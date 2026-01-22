            break;
        }

        case DEMUX_GET_PTS_DELAY:
            *va_arg (args, int64_t *) = 1000 * INT64_C(1000);
            break;

        default:
            return VLC_EGENERIC;
    }
    return VLC_SUCCESS;
}

void PlaylistManager::setBufferingRunState(bool b)
{
    vlc_mutex_lock(&lock);
    b_buffering = b;
    vlc_cond_signal(&waitcond);
    vlc_mutex_unlock(&lock);
}

void PlaylistManager::Run()
{
    vlc_mutex_lock(&lock);
    const unsigned i_min_buffering = playlist->getMinBuffering();
    const unsigned i_extra_buffering = playlist->getMaxBuffering() - i_min_buffering;
    while(1)
    {
        mutex_cleanup_push(&lock);
        while(!b_buffering)
            vlc_cond_wait(&waitcond, &lock);
        vlc_testcancel();
        vlc_cleanup_pop();

        if(needsUpdate())
        {
            int canc = vlc_savecancel();
            if(updatePlaylist())
                scheduleNextUpdate();
            else
                failedupdates++;
            vlc_restorecancel(canc);
        }

        vlc_mutex_lock(&demux.lock);
        mtime_t i_nzpcr = demux.i_nzpcr;
        vlc_mutex_unlock(&demux.lock);

        int canc = vlc_savecancel();
        AbstractStream::buffering_status i_return = bufferize(i_nzpcr, i_min_buffering, i_extra_buffering);
        vlc_restorecancel( canc );

        if(i_return != AbstractStream::buffering_lessthanmin)
        {
            mtime_t i_deadline = mdate();
            if(i_return == AbstractStream::buffering_ongoing)
