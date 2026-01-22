    *value = p;
    trim_right(*value);

    return 0;
}

static int parse_multipart_header(AVIOContext *pb,
                                    int* size,
                                    const char* expected_boundary,
                                    void *log_ctx);

static int mpjpeg_read_close(AVFormatContext *s)
{
    MPJPEGDemuxContext *mpjpeg = s->priv_data;
    av_freep(&mpjpeg->boundary);
