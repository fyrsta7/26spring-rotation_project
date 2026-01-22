                      p_buffer->i_pts - prev_date );

            aout_buffer_t *p_deleted;
            while( (p_deleted = p_fifo->p_first) != p_buffer )
                aout_BufferFree( aout_FifoPop( p_fifo ) );
        }

        prev_date = p_buffer->i_pts + p_buffer->i_length;
    }

    if( !AOUT_FMT_NON_LINEAR( &p_aout->format ) )
    {
        p_buffer = p_fifo->p_first;

        /* Additionally check that p_first_byte_to_mix is well located. */
        const unsigned framesize = p_aout->format.i_bytes_per_frame;
        ssize_t delta = (start_date - p_buffer->i_pts)
                      * p_aout->format.i_rate / CLOCK_FREQ;
        if( delta != 0 )
            msg_Warn( p_aout, "input start is not output end (%zd)", delta );
        if( delta < 0 )
        {
            /* Is it really the best way to do it ? */
            aout_FifoReset (&p->fifo);
            return NULL;
        }
        if( delta > 0 )
        {
            mtime_t t = delta * CLOCK_FREQ / p_aout->format.i_rate;
            p_buffer->i_nb_samples -= delta;
            p_buffer->i_pts += t;
            p_buffer->i_length -= t;
            delta *= framesize;
            p_buffer->p_buffer += delta;
            p_buffer->i_buffer -= delta;
        }

        /* Build packet with adequate number of samples */
        unsigned needed = samples * framesize;
        p_buffer = block_Alloc( needed );
        if( unlikely(p_buffer == NULL) )
            /* XXX: should free input buffers */
            return NULL;
        p_buffer->i_nb_samples = samples;

        for( uint8_t *p_out = p_buffer->p_buffer; needed > 0; )
        {
            aout_buffer_t *p_inbuf = p_fifo->p_first;
            if( unlikely(p_inbuf == NULL) )
            {
                msg_Err( p_aout, "packetization error" );
                vlc_memset( p_out, 0, needed );
                break;
            }

            const uint8_t *p_in = p_inbuf->p_buffer;
            size_t avail = p_inbuf->i_nb_samples * framesize;
            if( avail > needed )
            {
                vlc_memcpy( p_out, p_in, needed );
                p_fifo->p_first->p_buffer += needed;
                p_fifo->p_first->i_buffer -= needed;
                needed /= framesize;
                p_fifo->p_first->i_nb_samples -= needed;

                mtime_t t = needed * CLOCK_FREQ / p_aout->format.i_rate;
                p_fifo->p_first->i_pts += t;
