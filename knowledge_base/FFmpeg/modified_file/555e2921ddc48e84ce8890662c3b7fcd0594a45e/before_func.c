        return header->max_score;

    header->max_score = FLAC_HEADER_BASE_SCORE;

    /* Check and compute the children's scores. */
    child = header->next;
    for (dist = 0; dist < FLAC_MAX_SEQUENTIAL_HEADERS && child; dist++) {
        /* Look at the child's frame header info and penalize suspicious
           changes between the headers. */
        if (header->link_penalty[dist] == FLAC_HEADER_NOT_PENALIZED_YET) {
            header->link_penalty[dist] = check_header_mismatch(fpc, header,
                                                               child, AV_LOG_DEBUG);
        }
        child_score = score_header(fpc, child) - header->link_penalty[dist];

        if (FLAC_HEADER_BASE_SCORE + child_score > header->max_score) {
            /* Keep the child because the frame scoring is dynamic. */
            header->best_child = child;
            header->max_score  = FLAC_HEADER_BASE_SCORE + child_score;
        }
        child = child->next;
    }

    return header->max_score;
}

static void score_sequences(FLACParseContext *fpc)
{
    FLACHeaderMarker *curr;
    int best_score = FLAC_HEADER_NOT_SCORED_YET;
    /* First pass to clear all old scores. */
    for (curr = fpc->headers; curr; curr = curr->next)
        curr->max_score = FLAC_HEADER_NOT_SCORED_YET;

    /* Do a second pass to score them all. */
    for (curr = fpc->headers; curr; curr = curr->next) {
        if (score_header(fpc, curr) > best_score) {
            fpc->best_header = curr;
            best_score       = curr->max_score;
        }
    }
}

static int get_best_header(FLACParseContext* fpc, const uint8_t **poutbuf,
                           int *poutbuf_size)
{
    FLACHeaderMarker *header = fpc->best_header;
    FLACHeaderMarker *child  = header->best_child;
    if (!child) {
        *poutbuf_size = av_fifo_size(fpc->fifo_buf) - header->offset;
    } else {
        *poutbuf_size = child->offset - header->offset;

        /* If the child has suspicious changes, log them */
        check_header_mismatch(fpc, header, child, 0);
    }

    fpc->avctx->sample_rate = header->fi.samplerate;
    fpc->avctx->channels    = header->fi.channels;
    fpc->pc->duration       = header->fi.blocksize;
    *poutbuf = flac_fifo_read_wrap(fpc, header->offset, *poutbuf_size,
                                        &fpc->wrap_buf,
                                        &fpc->wrap_buf_allocated_size);

    fpc->best_header_valid = 0;
    /* Return the negative overread index so the client can compute pos.
       This should be the amount overread to the beginning of the child */
    if (child)
        return child->offset - av_fifo_size(fpc->fifo_buf);
    return 0;
}

static int flac_parse(AVCodecParserContext *s, AVCodecContext *avctx,
                      const uint8_t **poutbuf, int *poutbuf_size,
                      const uint8_t *buf, int buf_size)
{
    FLACParseContext *fpc = s->priv_data;
    FLACHeaderMarker *curr;
    int nb_headers;
    const uint8_t *read_end   = buf;
    const uint8_t *read_start = buf;

    if (s->flags & PARSER_FLAG_COMPLETE_FRAMES) {
        FLACFrameInfo fi;
        if (frame_header_is_valid(avctx, buf, &fi))
            s->duration = fi.blocksize;
        *poutbuf      = buf;
        *poutbuf_size = buf_size;
        return buf_size;
    }

    fpc->avctx = avctx;
    if (fpc->best_header_valid)
        return get_best_header(fpc, poutbuf, poutbuf_size);

    /* If a best_header was found last call remove it with the buffer data. */
    if (fpc->best_header && fpc->best_header->best_child) {
        FLACHeaderMarker *temp;
        FLACHeaderMarker *best_child = fpc->best_header->best_child;

        /* Remove headers in list until the end of the best_header. */
        for (curr = fpc->headers; curr != best_child; curr = temp) {
            if (curr != fpc->best_header) {
                av_log(avctx, AV_LOG_DEBUG,
                       "dropping low score %i frame header from offset %i to %i\n",
                       curr->max_score, curr->offset, curr->next->offset);
            }
            temp = curr->next;
            av_freep(&curr->link_penalty);
            av_free(curr);
            fpc->nb_headers_buffered--;
        }
        /* Release returned data from ring buffer. */
        av_fifo_drain(fpc->fifo_buf, best_child->offset);

        /* Fix the offset for the headers remaining to match the new buffer. */
        for (curr = best_child->next; curr; curr = curr->next)
            curr->offset -= best_child->offset;

        fpc->nb_headers_buffered--;
        best_child->offset = 0;
        fpc->headers       = best_child;
        if (fpc->nb_headers_buffered >= FLAC_MIN_HEADERS) {
            fpc->best_header = best_child;
            return get_best_header(fpc, poutbuf, poutbuf_size);
        }
        fpc->best_header   = NULL;
    } else if (fpc->best_header) {
        /* No end frame no need to delete the buffer; probably eof */
        FLACHeaderMarker *temp;

        for (curr = fpc->headers; curr != fpc->best_header; curr = temp) {
            temp = curr->next;
            av_freep(&curr->link_penalty);
            av_free(curr);
        }
        fpc->headers = fpc->best_header->next;
        av_freep(&fpc->best_header->link_penalty);
        av_freep(&fpc->best_header);
    }

    /* Find and score new headers. */
    while ((buf && read_end < buf + buf_size &&
            fpc->nb_headers_buffered < FLAC_MIN_HEADERS)
           || (!buf && !fpc->end_padded)) {
        int start_offset;

        /* Pad the end once if EOF, to check the final region for headers. */
        if (!buf) {
            fpc->end_padded      = 1;
            buf_size = MAX_FRAME_HEADER_SIZE;
            read_end = read_start + MAX_FRAME_HEADER_SIZE;
        } else {
            /* The maximum read size is the upper-bound of what the parser
               needs to have the required number of frames buffered */
            int nb_desired = FLAC_MIN_HEADERS - fpc->nb_headers_buffered + 1;
            read_end       = read_end + FFMIN(buf + buf_size - read_end,
                                              nb_desired * FLAC_AVG_FRAME_SIZE);
        }

        /* Fill the buffer. */
        if (av_fifo_realloc2(fpc->fifo_buf,
                             (read_end - read_start) + av_fifo_size(fpc->fifo_buf)) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "couldn't reallocate buffer of size %td\n",
                   (read_end - read_start) + av_fifo_size(fpc->fifo_buf));
            goto handle_error;
        }

        if (buf) {
            av_fifo_generic_write(fpc->fifo_buf, (void*) read_start,
                                  read_end - read_start, NULL);
        } else {
            int8_t pad[MAX_FRAME_HEADER_SIZE] = { 0 };
            av_fifo_generic_write(fpc->fifo_buf, (void*) pad, sizeof(pad), NULL);
