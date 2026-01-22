}

vlc_tick_t AbstractStream::getDemuxedAmount(vlc_tick_t from) const
{
    return fakeEsOut()->commandsQueue()->getDemuxedAmount(from);
}

AbstractStream::BufferingStatus AbstractStream::bufferize(vlc_tick_t nz_deadline,
                                                           vlc_tick_t i_min_buffering,
                                                           vlc_tick_t i_extra_buffering,
                                                           vlc_tick_t i_target_buffering,
                                                           bool b_keep_alive)
{
    last_buffer_status = doBufferize(nz_deadline, i_min_buffering, i_extra_buffering,
                                     i_target_buffering, b_keep_alive);
    return last_buffer_status;
}

AbstractStream::BufferingStatus AbstractStream::doBufferize(vlc_tick_t nz_deadline,
                                                             vlc_tick_t i_min_buffering,
                                                             vlc_tick_t i_max_buffering,
                                                             vlc_tick_t i_target_buffering,
                                                             bool b_keep_alive)
{
    vlc_mutex_lock(&lock);

    /* Ensure it is configured */
    if(!segmentTracker || !connManager || !valid)
    {
        vlc_mutex_unlock(&lock);
        return BufferingStatus::End;
    }

    /* Disable streams that are not selected (alternate streams) */
    if(esCount() && !isSelected() && !fakeEsOut()->restarting() && !b_keep_alive)
    {
        setDisabled(true);
        segmentTracker->reset();
        fakeEsOut()->commandsQueue()->Abort(false);
        msg_Dbg(p_realdemux, "deactivating %s stream %s",
                format.str().c_str(), description.c_str());
        vlc_mutex_unlock(&lock);
        return BufferingStatus::End;
    }

    if(fakeEsOut()->commandsQueue()->isDraining())
    {
        vlc_mutex_unlock(&lock);
        return BufferingStatus::Suspended;
    }

    segmentTracker->setStartPosition();

    /* Reached end of live playlist */
    if(!segmentTracker->bufferingAvailable())
    {
        vlc_mutex_unlock(&lock);
        return BufferingStatus::Suspended;
    }

    if(!demuxer)
    {
        if(!startDemux())
        {
            valid = false; /* Prevent further retries */
            fakeEsOut()->commandsQueue()->setEOF(true);
            vlc_mutex_unlock(&lock);
            return BufferingStatus::End;
        }
    }

    vlc_tick_t i_demuxed = fakeEsOut()->commandsQueue()->getDemuxedAmount(nz_deadline);
    segmentTracker->notifyBufferingLevel(i_min_buffering, i_max_buffering, i_demuxed, i_target_buffering);
    if(i_demuxed < i_max_buffering) /* not already demuxed */
    {
        vlc_tick_t nz_extdeadline = fakeEsOut()->commandsQueue()->getBufferingLevel() +
                                    (i_max_buffering - i_demuxed) / 4;
        nz_deadline = std::max(nz_deadline, nz_extdeadline);

        /* need to read, demuxer still buffering, ... */
        vlc_mutex_unlock(&lock);
        Demuxer::Status demuxStatus = demuxer->demux(nz_deadline);
        vlc_mutex_lock(&lock);
        if(demuxStatus != Demuxer::Status::Success)
        {
            if(discontinuity || needrestart)
            {
                msg_Dbg(p_realdemux, "Restarting demuxer %d %d", needrestart, discontinuity);
                prepareRestart(discontinuity);
                if(discontinuity)
                {
                    msg_Dbg(p_realdemux, "Draining on discontinuity");
                    fakeEsOut()->commandsQueue()->setDraining();
                    discontinuity = false;
                }
                needrestart = false;
                vlc_mutex_unlock(&lock);
                return BufferingStatus::Ongoing;
