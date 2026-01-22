        ret = av_frame_ref(ost->sq_frame, frame);
        if (ret < 0)
            return ret;
        frame = ost->sq_frame;
    }

    ret = sq_send(of->sq_encode, ost->sq_idx_encode,
                  SQFRAME(frame));
    if (ret < 0) {
        if (frame)
            av_frame_unref(frame);
        if (ret != AVERROR_EOF)
            return ret;
    }

    while (1) {
        AVFrame *enc_frame = ost->sq_frame;

        ret = sq_receive(of->sq_encode, ost->sq_idx_encode,
                               SQFRAME(enc_frame));
        if (ret == AVERROR_EOF) {
            enc_frame = NULL;
        } else if (ret < 0) {
            return (ret == AVERROR(EAGAIN)) ? 0 : ret;
        }
