    SrsQueueRecvThread trd(consumer, rtmp, SRS_PERF_MW_SLEEP, _srs_context->get_id());
    
    if ((err = trd.start()) != srs_success) {
        return srs_error_wrap(err, "rtmp: start receive thread");
    }
    
    // Deliver packets to peer.
    wakable = consumer;
    err = do_playing(source, consumer, &trd);
    wakable = NULL;
    
    trd.stop();
    
    // Drop all packets in receiving thread.
    if (!trd.empty()) {
        srs_warn("drop the received %d messages", trd.size());
    }
    
    return err;
}

srs_error_t SrsRtmpConn::do_playing(SrsSource* source, SrsConsumer* consumer, SrsQueueRecvThread* rtrd)
{
    srs_error_t err = srs_success;
    
    SrsRequest* req = info->req;
    srs_assert(req);
    srs_assert(consumer);
    
    // initialize other components
    SrsPithyPrint* pprint = SrsPithyPrint::create_rtmp_play();
    SrsAutoFree(SrsPithyPrint, pprint);
    
    SrsMessageArray msgs(SRS_PERF_MW_MSGS);
    bool user_specified_duration_to_stop = (req->duration > 0);
    int64_t starttime = -1;

    // setup the realtime.
    realtime = _srs_config->get_realtime_enabled(req->vhost);
    // setup the mw config.
    // when mw_sleep changed, resize the socket send buffer.
    mw_msgs = _srs_config->get_mw_msgs(req->vhost, realtime);
    mw_sleep = _srs_config->get_mw_sleep(req->vhost);
    skt->set_socket_buffer(mw_sleep);
    // initialize the send_min_interval
    send_min_interval = _srs_config->get_send_min_interval(req->vhost);
    
    srs_trace("start play smi=%dms, mw_sleep=%d, mw_msgs=%d, realtime=%d, tcp_nodelay=%d",
        srsu2msi(send_min_interval), srsu2msi(mw_sleep), mw_msgs, realtime, tcp_nodelay);
    
    while (true) {
        // when source is set to expired, disconnect it.
        if ((err = trd->pull()) != srs_success) {
            return srs_error_wrap(err, "rtmp: thread quit");
        }

        // collect elapse for pithy print.
        pprint->elapse();

        // to use isolate thread to recv, can improve about 33% performance.
        // @see: https://github.com/ossrs/srs/issues/196
        // @see: https://github.com/ossrs/srs/issues/217
        while (!rtrd->empty()) {
            SrsCommonMessage* msg = rtrd->pump();
            if ((err = process_play_control_msg(consumer, msg)) != srs_success) {
                return srs_error_wrap(err, "rtmp: play control message");
            }
        }
        
        // quit when recv thread error.
        if ((err = rtrd->error_code()) != srs_success) {
            return srs_error_wrap(err, "rtmp: recv thread");
        }
        
#ifdef SRS_PERF_QUEUE_COND_WAIT
        // wait for message to incoming.
        // @see https://github.com/ossrs/srs/issues/251
        // @see https://github.com/ossrs/srs/issues/257
        consumer->wait(mw_msgs, mw_sleep);
#endif
        
        // get messages from consumer.
        // each msg in msgs.msgs must be free, for the SrsMessageArray never free them.
        // @remark when enable send_min_interval, only fetch one message a time.
        int count = (send_min_interval > 0)? 1 : 0;
        if ((err = consumer->dump_packets(&msgs, count)) != srs_success) {
            return srs_error_wrap(err, "rtmp: consumer dump packets");
        }

        // reportable
        if (pprint->can_print()) {
            kbps->sample();
            srs_trace("-> " SRS_CONSTS_LOG_PLAY " time=%d, msgs=%d, okbps=%d,%d,%d, ikbps=%d,%d,%d, mw=%d/%d",
                (int)pprint->age(), count, kbps->get_send_kbps(), kbps->get_send_kbps_30s(), kbps->get_send_kbps_5m(),
                kbps->get_recv_kbps(), kbps->get_recv_kbps_30s(), kbps->get_recv_kbps_5m(), srsu2msi(mw_sleep), mw_msgs);
        }
        
        if (count <= 0) {
#ifndef SRS_PERF_QUEUE_COND_WAIT
            srs_usleep(mw_sleep);
#endif
            // ignore when nothing got.
            continue;
        }
        
        // only when user specifies the duration,
        // we start to collect the durations for each message.
        if (user_specified_duration_to_stop) {
            for (int i = 0; i < count; i++) {
                SrsSharedPtrMessage* msg = msgs.msgs[i];
                
                // foreach msg, collect the duration.
                // @remark: never use msg when sent it, for the protocol sdk will free it.
                if (starttime < 0 || starttime > msg->timestamp) {
                    starttime = msg->timestamp;
                }
                duration += (msg->timestamp - starttime) * SRS_UTIME_MILLISECONDS;
                starttime = msg->timestamp;
            }
        }
        
        // sendout messages, all messages are freed by send_and_free_messages().
        // no need to assert msg, for the rtmp will assert it.
        if (count > 0 && (err = rtmp->send_and_free_messages(msgs.msgs, count, info->res->stream_id)) != srs_success) {
            return srs_error_wrap(err, "rtmp: send %d messages", count);
