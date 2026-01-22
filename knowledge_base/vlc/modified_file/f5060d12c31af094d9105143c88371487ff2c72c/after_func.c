                uint32_t u;
                uint8_t a[4];
            } c;
            c.a[0] = p_dec->fmt_in.video.p_palette->palette[i][0];
            c.a[1] = p_dec->fmt_in.video.p_palette->palette[i][1];
            c.a[2] = p_dec->fmt_in.video.p_palette->palette[i][2];
            c.a[3] = p_dec->fmt_in.video.p_palette->palette[i][3];

            p_sys->palette.palette[i] = c.u;
        }
        p_sys->p_context->palctrl = &p_sys->palette;

        p_dec->fmt_out.video.p_palette = malloc( sizeof(video_palette_t) );
        if( p_dec->fmt_out.video.p_palette )
            *p_dec->fmt_out.video.p_palette = *p_dec->fmt_in.video.p_palette;
    }
    else if( p_sys->i_codec_id != CODEC_ID_MSVIDEO1 && p_sys->i_codec_id != CODEC_ID_CINEPAK )
    {
        p_sys->p_context->palctrl = &p_sys->palette;
    }

    /* ***** init this codec with special data ***** */
    ffmpeg_InitCodec( p_dec );

    /* ***** Open the codec ***** */
    if( ffmpeg_OpenCodec( p_dec ) < 0 )
    {
        msg_Err( p_dec, "cannot open codec (%s)", p_sys->psz_namecodec );
        free( p_sys->p_buffer_orig );
        free( p_sys );
        return VLC_EGENERIC;
    }

    return VLC_SUCCESS;
}

/*****************************************************************************
 * DecodeVideo: Called to decode one or more frames
 *****************************************************************************/
picture_t *DecodeVideo( decoder_t *p_dec, block_t **pp_block )
{
    decoder_sys_t *p_sys = p_dec->p_sys;
    int b_drawpicture;
    int b_null_size = false;
    block_t *p_block;

    if( !pp_block || !*pp_block )
        return NULL;

    if( !p_sys->p_context->extradata_size && p_dec->fmt_in.i_extra )
    {
        ffmpeg_InitCodec( p_dec );
        if( p_sys->b_delayed_open )
        {
            if( ffmpeg_OpenCodec( p_dec ) )
                msg_Err( p_dec, "cannot open codec (%s)", p_sys->psz_namecodec );
        }
    }

    p_block = *pp_block;
    if( p_sys->b_delayed_open )
    {
        block_Release( p_block );
        return NULL;
    }

    if( p_block->i_flags & (BLOCK_FLAG_DISCONTINUITY|BLOCK_FLAG_CORRUPTED) )
    {
        p_sys->i_buffer = 0;
        p_sys->i_pts = 0; /* To make sure we recover properly */

        p_sys->input_pts = p_sys->input_dts = 0;
        p_sys->i_late_frames = 0;

        block_Release( p_block );

        //if( p_block->i_flags & BLOCK_FLAG_CORRUPTED )
            //avcodec_flush_buffers( p_sys->p_context );
        return NULL;
    }

    if( p_block->i_flags & BLOCK_FLAG_PREROLL )
    {
        /* Do not care about late frames when prerolling
         * TODO avoid decoding of non reference frame
         * (ie all B except for H264 where it depends only on nal_ref_idc) */
        p_sys->i_late_frames = 0;
    }

    if( !p_dec->b_pace_control && (p_sys->i_late_frames > 0) &&
        (mdate() - p_sys->i_late_frames_start > INT64_C(5000000)) )
    {
        if( p_sys->i_pts )
        {
            msg_Err( p_dec, "more than 5 seconds of late video -> "
                     "dropping frame (computer too slow ?)" );
            p_sys->i_pts = 0; /* To make sure we recover properly */
        }
        block_Release( p_block );
        p_sys->i_late_frames--;
        return NULL;
    }

    if( p_block->i_pts > 0 || p_block->i_dts > 0 )
    {
        p_sys->input_pts = p_block->i_pts;
        p_sys->input_dts = p_block->i_dts;

        /* Make sure we don't reuse the same timestamps twice */
        p_block->i_pts = p_block->i_dts = 0;
    }

    /* A good idea could be to decode all I pictures and see for the other */
    if( !p_dec->b_pace_control &&
        p_sys->b_hurry_up &&
        (p_sys->i_late_frames > 4) )
    {
        b_drawpicture = 0;
        if( p_sys->i_late_frames < 12 )
        {
            p_sys->p_context->skip_frame =
                    (p_sys->i_skip_frame <= AVDISCARD_BIDIR) ?
                    AVDISCARD_BIDIR : p_sys->i_skip_frame;
        }
        else
        {
            /* picture too late, won't decode
             * but break picture until a new I, and for mpeg4 ...*/
            p_sys->i_late_frames--; /* needed else it will never be decrease */
            block_Release( p_block );
            p_sys->i_buffer = 0;
            return NULL;
        }
    }
    else
    {
        if( p_sys->b_hurry_up )
            p_sys->p_context->skip_frame = p_sys->i_skip_frame;
        if( !(p_block->i_flags & BLOCK_FLAG_PREROLL) )
            b_drawpicture = 1;
        else
            b_drawpicture = 0;
    }

    if( p_sys->p_context->width <= 0 || p_sys->p_context->height <= 0 )
    {
        if( p_sys->b_hurry_up )
            p_sys->p_context->skip_frame = p_sys->i_skip_frame;
        b_null_size = true;
    }
    else if( !b_drawpicture )
    {
        p_sys->p_context->skip_frame = __MAX( p_sys->p_context->skip_frame,
                                              AVDISCARD_NONREF );
    }

    /*
     * Do the actual decoding now
     */

    /* Don't forget that ffmpeg requires a little more bytes
     * that the real frame size */
    if( p_block->i_buffer > 0 )
    {
        p_sys->b_flush = ( p_block->i_flags & BLOCK_FLAG_END_OF_SEQUENCE ) != 0;

        p_sys->i_buffer = p_block->i_buffer;
        if( p_sys->i_buffer + FF_INPUT_BUFFER_PADDING_SIZE >
            p_sys->i_buffer_orig )
        {
            free( p_sys->p_buffer_orig );
            p_sys->i_buffer_orig =
                p_block->i_buffer + FF_INPUT_BUFFER_PADDING_SIZE;
            p_sys->p_buffer_orig = malloc( p_sys->i_buffer_orig );
        }
        p_sys->p_buffer = p_sys->p_buffer_orig;
        p_sys->i_buffer = p_block->i_buffer;
        if( !p_sys->p_buffer )
        {
            block_Release( p_block );
            return NULL;
        }
        vlc_memcpy( p_sys->p_buffer, p_block->p_buffer, p_block->i_buffer );
        memset( p_sys->p_buffer + p_block->i_buffer, 0,
                FF_INPUT_BUFFER_PADDING_SIZE );

        p_block->i_buffer = 0;
    }

    while( p_sys->i_buffer > 0 || p_sys->b_flush )
    {
        int i_used, b_gotpicture;
        picture_t *p_pic;

        i_used = avcodec_decode_video( p_sys->p_context, p_sys->p_ff_pic,
                                       &b_gotpicture,
                                       p_sys->i_buffer <= 0 && p_sys->b_flush ? NULL : (uint8_t*)p_sys->p_buffer, p_sys->i_buffer );

        if( b_null_size && p_sys->p_context->width > 0 &&
            p_sys->p_context->height > 0 &&
            !p_sys->b_flush )
        {
            /* Reparse it to not drop the I frame */
            b_null_size = false;
            if( p_sys->b_hurry_up )
                p_sys->p_context->skip_frame = p_sys->i_skip_frame;
            i_used = avcodec_decode_video( p_sys->p_context, p_sys->p_ff_pic,
                                           &b_gotpicture,
                                           (uint8_t*)p_sys->p_buffer, p_sys->i_buffer );
        }

        if( p_sys->b_flush )
            p_sys->b_first_frame = true;

        if( p_sys->i_buffer <= 0 )
            p_sys->b_flush = false;

        if( i_used < 0 )
        {
            if( b_drawpicture )
                msg_Warn( p_dec, "cannot decode one frame (%d bytes)",
                          p_sys->i_buffer );
            block_Release( p_block );
            return NULL;
        }
        else if( i_used > p_sys->i_buffer )
        {
            i_used = p_sys->i_buffer;
        }

        /* Consumed bytes */
        p_sys->i_buffer -= i_used;
        p_sys->p_buffer += i_used;

        /* Nothing to display */
        if( !b_gotpicture )
        {
            if( i_used == 0 ) break;
            continue;
        }

        /* Set the PTS */
        if( p_sys->p_ff_pic->pts )
            p_sys->i_pts = p_sys->p_ff_pic->pts;

        /* Update frame late count (except when doing preroll) */
        mtime_t i_display_date = 0;
        if( !(p_block->i_flags & BLOCK_FLAG_PREROLL) )
            i_display_date = decoder_GetDisplayDate( p_dec, p_sys->i_pts );

        if( i_display_date > 0 && i_display_date <= mdate() )
        {
            p_sys->i_late_frames++;
            if( p_sys->i_late_frames == 1 )
                p_sys->i_late_frames_start = mdate();
        }
        else
        {
            p_sys->i_late_frames = 0;
        }

        if( !b_drawpicture || !p_sys->p_ff_pic->linesize[0] )
        {
            /* Do not display the picture */
            p_pic = (picture_t *)p_sys->p_ff_pic->opaque;
            if( !b_drawpicture && p_pic )
                decoder_DeletePicture( p_dec, p_pic );

            ffmpeg_NextPts( p_dec );
            continue;
        }

        if( !p_sys->p_ff_pic->opaque )
        {
            /* Get a new picture */
            p_pic = ffmpeg_NewPictBuf( p_dec, p_sys->p_context );
            if( !p_pic )
            {
                block_Release( p_block );
                return NULL;
            }

            /* Fill p_picture_t from AVVideoFrame and do chroma conversion
             * if needed */
            ffmpeg_CopyPicture( p_dec, p_pic, p_sys->p_ff_pic );
        }
        else
        {
            p_pic = (picture_t *)p_sys->p_ff_pic->opaque;
        }

        /* Sanity check (seems to be needed for some streams) */
        if( p_sys->p_ff_pic->pict_type == FF_B_TYPE )
        {
            p_sys->b_has_b_frames = true;
        }

        if( !p_dec->fmt_in.video.i_aspect )
        {
            /* Fetch again the aspect ratio in case it changed */
            p_dec->fmt_out.video.i_aspect =
                VOUT_ASPECT_FACTOR
                    * ( av_q2d(p_sys->p_context->sample_aspect_ratio)
                    * p_sys->p_context->width / p_sys->p_context->height );
            p_dec->fmt_out.video.i_sar_num
                = p_sys->p_context->sample_aspect_ratio.num;
            p_dec->fmt_out.video.i_sar_den
                = p_sys->p_context->sample_aspect_ratio.den;

            if( p_dec->fmt_out.video.i_aspect == 0 )
            {
                p_dec->fmt_out.video.i_aspect = VOUT_ASPECT_FACTOR
                    * p_sys->p_context->width / p_sys->p_context->height;
            }
        }

        /* Send decoded frame to vout */
