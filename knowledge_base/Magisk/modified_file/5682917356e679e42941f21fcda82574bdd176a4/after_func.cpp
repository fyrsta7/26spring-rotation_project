    gz_strm(mode_t mode, stream_ptr &&base) :
        cpr_stream(std::move(base)), mode(mode), strm{}, outbuf{0} {
        switch(mode) {
            case DECODE:
                inflateInit2(&strm, 15 | 16);
                break;
            case ENCODE:
                deflateInit2(&strm, 9, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY);
                break;
        }
    }
