                omx_error, (int)p_port->i_port_index );

    omx_error = WaitForSpecificOmxEvent(&p_sys->event_queue, OMX_EventCmdComplete, 0, 0, 0);
    CHECK_ERROR(omx_error, "Wait for PortEnable failed (%x)", omx_error );

    PrintOmx(p_dec, p_sys->omx_handle, p_dec->p_sys->in.i_port_index);
    PrintOmx(p_dec, p_sys->omx_handle, p_dec->p_sys->out.i_port_index);

 error:
    return omx_error;
}

/*****************************************************************************
 * DecodeVideoOutput
 *****************************************************************************/
static int DecodeVideoOutput( decoder_t *p_dec, OmxPort *p_port, picture_t **pp_pic )
{
    VLC_UNUSED( p_dec );
    OMX_BUFFERHEADERTYPE *p_header;
    picture_t *p_pic = NULL, *p_next_pic;
    OMX_ERRORTYPE omx_error;

    while(!p_pic)
    {
        OMX_FIFO_PEEK(&p_port->fifo, p_header);
        if(!p_header) break; /* No frame available */

        if(p_port->b_update_def)
        {
            omx_error = GetPortDefinition(p_dec, p_port, p_port->p_fmt);
            p_port->b_update_def = 0;
            CHECK_ERROR(omx_error, "GetPortDefinition failed");
        }

        if(p_header->nFilledLen)
        {
            p_pic = p_header->pAppPrivate;
            if(!p_pic)
            {
                /* We're not in direct rendering mode.
                 * Get a new picture and copy the content */
                p_pic = decoder_NewPicture( p_dec );

                if (p_pic)
                    CopyOmxPicture(p_port->definition.format.video.eColorFormat,
                                   p_pic, p_port->definition.format.video.nSliceHeight,
                                   p_port->i_frame_stride,
                                   p_header->pBuffer + p_header->nOffset,
                                   p_port->i_frame_stride_chroma_div, NULL);
            }

            if (p_pic)
                p_pic->date = FromOmxTicks(p_header->nTimeStamp);
            p_header->nFilledLen = 0;
            p_header->pAppPrivate = 0;
        }

        /* Get a new picture */
        if(p_port->b_direct && !p_header->pAppPrivate)
        {
            p_next_pic = decoder_NewPicture( p_dec );
            if(!p_next_pic) break;

            OMX_FIFO_GET(&p_port->fifo, p_header);
            p_header->pAppPrivate = p_next_pic;
            p_header->pInputPortPrivate = p_header->pBuffer;
            p_header->pBuffer = p_next_pic->p[0].p_pixels;
        }
        else
        {
            OMX_FIFO_GET(&p_port->fifo, p_header);
        }

#ifdef OMXIL_EXTRA_DEBUG
        msg_Dbg( p_dec, "FillThisBuffer %p, %p", p_header, p_header->pBuffer );
#endif
        OMX_FillThisBuffer(p_port->omx_handle, p_header);
    }

    *pp_pic = p_pic;
    return 0;
error:
    return -1;
}

/*****************************************************************************
 * DecodeVideo: Called to decode one frame
 *****************************************************************************/
static picture_t *DecodeVideo( decoder_t *p_dec, block_t **pp_block )
{
    decoder_sys_t *p_sys = p_dec->p_sys;
    picture_t *p_pic = NULL;
    OMX_ERRORTYPE omx_error;
    unsigned int i;

    OMX_BUFFERHEADERTYPE *p_header;
    block_t *p_block;
    unsigned int i_input_used = 0;
    struct H264ConvertState convert_state = { 0, 0 };

    if( !pp_block || !*pp_block )
        return NULL;

    p_block = *pp_block;

    /* Check for errors from codec */
    if(p_sys->b_error)
    {
        msg_Dbg(p_dec, "error during decoding");
        block_Release( p_block );
        return 0;
    }

    if( p_block->i_flags & (BLOCK_FLAG_DISCONTINUITY|BLOCK_FLAG_CORRUPTED) )
    {
        block_Release( p_block );
        if(!p_sys->in.b_flushed)
        {
            msg_Dbg(p_dec, "flushing");
            OMX_SendCommand( p_sys->omx_handle, OMX_CommandFlush,
                             p_sys->in.definition.nPortIndex, 0 );
        }
        p_sys->in.b_flushed = true;
        return NULL;
    }

    /* Use the aspect ratio provided by the input (ie read from packetizer).
     * In case the we get aspect ratio info from the decoder (as in the
     * broadcom OMX implementation on RPi), don't let the packetizer values
     * override what the decoder says, if anything - otherwise always update
     * even if it already is set (since it can change within a stream). */
    if((p_dec->fmt_in.video.i_sar_num != 0 && p_dec->fmt_in.video.i_sar_den != 0) &&
       (p_dec->fmt_out.video.i_sar_num == 0 || p_dec->fmt_out.video.i_sar_den == 0 ||
             !p_sys->b_aspect_ratio_handled))
    {
        p_dec->fmt_out.video.i_sar_num = p_dec->fmt_in.video.i_sar_num;
        p_dec->fmt_out.video.i_sar_den = p_dec->fmt_in.video.i_sar_den;
