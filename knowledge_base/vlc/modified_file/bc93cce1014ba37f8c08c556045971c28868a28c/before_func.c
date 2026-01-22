                p_sys->i_last_pic %= PICTURE_RING_SIZE;
            }
            vlc_cond_signal( &p_sys->cond );
            vlc_mutex_unlock( &p_sys->lock_out );
        }
    }

    return VLC_SUCCESS;
}

static void* EncoderThread( vlc_object_t* p_this )
{
    sout_stream_sys_t *p_sys = (sout_stream_sys_t*)p_this;
    sout_stream_id_t *id = p_sys->id_video;
    picture_t *p_pic;
    int canc = vlc_savecancel ();

    while( vlc_object_alive (p_sys) && !p_sys->b_error )
    {
        block_t *p_block;

