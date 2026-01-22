    int default_rcvbuf = 0;
    // TODO: FIXME: Config it.
    int expect_rcvbuf = 1024*1024*10; // 10M
    int actual_rcvbuf = expect_rcvbuf;
    int r0_rcvbuf = 0;
    if (true) {
        socklen_t opt_len = sizeof(default_rcvbuf);
        getsockopt(fd(), SOL_SOCKET, SO_RCVBUF, (void*)&default_rcvbuf, &opt_len);

        if ((r0_rcvbuf = setsockopt(fd(), SOL_SOCKET, SO_RCVBUF, (void*)&actual_rcvbuf, sizeof(actual_rcvbuf))) < 0) {
            srs_warn("set SO_RCVBUF failed, expect=%d, r0=%d", expect_rcvbuf, r0_rcvbuf);
        }

        opt_len = sizeof(actual_rcvbuf);
        getsockopt(fd(), SOL_SOCKET, SO_RCVBUF, (void*)&actual_rcvbuf, &opt_len);
    }

    srs_trace("UDP #%d LISTEN at %s:%d, SO_SNDBUF(default=%d, expect=%d, actual=%d, r0=%d), SO_RCVBUF(default=%d, expect=%d, actual=%d, r0=%d)",
        srs_netfd_fileno(lfd), ip.c_str(), port, default_sndbuf, expect_sndbuf, actual_sndbuf, r0_sndbuf, default_rcvbuf, expect_rcvbuf, actual_rcvbuf, r0_rcvbuf);
}

srs_error_t SrsUdpMuxListener::cycle()
{
    srs_error_t err = srs_success;

    SrsPithyPrint* pprint = SrsPithyPrint::create_rtc_recv(srs_netfd_fileno(lfd));
    SrsAutoFree(SrsPithyPrint, pprint);

    uint64_t nn_msgs = 0;
    uint64_t nn_msgs_stage = 0;
    uint64_t nn_msgs_last = 0;
    uint64_t nn_loop = 0;
    srs_utime_t time_last = srs_get_system_time();

    SrsErrorPithyPrint* pp_pkt_handler_err = new SrsErrorPithyPrint();
    SrsAutoFree(SrsErrorPithyPrint, pp_pkt_handler_err);

    set_socket_buffer();
    
    while (true) {
        if ((err = trd->pull()) != srs_success) {
            return srs_error_wrap(err, "udp listener");
        }

        nn_loop++;

        // TODO: FIXME: Refactor the memory cache for receiver.
        // Because we have to decrypt the cipher of received packet payload,
        // and the size is not determined, so we think there is at least one copy,
        // and we can reuse the plaintext h264/opus with players when got plaintext.
        SrsUdpMuxSocket skt(lfd);

        int nread = skt.recvfrom(SRS_UTIME_NO_TIMEOUT);
        if (nread <= 0) {
            if (nread < 0) {
                srs_warn("udp recv error nn=%d", nread);
            }
            // remux udp never return
            continue;
        }

        nn_msgs++;
        nn_msgs_stage++;

        // Handle the UDP packet.
        err = handler->on_udp_packet(&skt);

        // Use pithy print to show more smart information.
        if (err != srs_success) {
            uint32_t nn = 0;
            if (pp_pkt_handler_err->can_print(err, &nn)) {
                // For performance, only restore context when output log.
                _srs_context->set_id(cid);

                // Append more information.
                err = srs_error_wrap(err, "size=%u, data=[%s]", skt.size(), srs_string_dumps_hex(skt.data(), skt.size(), 8).c_str());
                srs_warn("handle udp pkt, count=%u/%u, err: %s", pp_pkt_handler_err->nn_count, nn, srs_error_desc(err).c_str());
            }
            srs_freep(err);
        }

        pprint->elapse();
        if (pprint->can_print()) {
            // For performance, only restore context when output log.
            _srs_context->set_id(cid);

            int pps_average = 0; int pps_last = 0;
            if (true) {
                if (srs_get_system_time() > srs_get_system_startup_time()) {
                    pps_average = (int)(nn_msgs * SRS_UTIME_SECONDS / (srs_get_system_time() - srs_get_system_startup_time()));
                }
                if (srs_get_system_time() > time_last) {
                    pps_last = (int)((nn_msgs - nn_msgs_last) * SRS_UTIME_SECONDS / (srs_get_system_time() - time_last));
                }
            }
