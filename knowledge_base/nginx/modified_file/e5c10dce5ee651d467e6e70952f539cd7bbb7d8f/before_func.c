    for (cl = frame->data; cl; cl = cl->next) {
        b = cl->buf;
        n = ngx_buf_size(b);

        if (n >= tail) {
            b->pos += tail;
            break;
        }

        cl->buf->pos = cl->buf->last;
        tail -= n;
    }

    return NGX_OK;
}


static ngx_int_t
ngx_quic_buffer_frame(ngx_connection_t *c, ngx_quic_frames_stream_t *fs,
    ngx_quic_frame_t *frame)
{
    ngx_queue_t               *q;
    ngx_quic_frame_t          *dst, *item;
    ngx_quic_ordered_frame_t  *f, *df;

    ngx_log_debug0(NGX_LOG_DEBUG_EVENT, c->log, 0,
                   "quic ngx_quic_buffer_frame");

    f = &frame->u.ord;

    /* frame start offset is in the future, buffer it */

    dst = ngx_quic_alloc_frame(c);
    if (dst == NULL) {
        return NGX_ERROR;
    }

    ngx_memcpy(dst, frame, sizeof(ngx_quic_frame_t));

    dst->data = ngx_quic_copy_chain(c, frame->data, 0);
    if (dst->data == NGX_CHAIN_ERROR) {
