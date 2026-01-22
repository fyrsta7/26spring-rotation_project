    { 0, NULL }
};

static int dashenc_io_open(AVFormatContext *s, AVIOContext **pb, char *filename,
                           AVDictionary **options) {
    DASHContext *c = s->priv_data;
    int http_base_proto = filename ? ff_is_http_proto(filename) : 0;
    int err = AVERROR_MUXER_NOT_FOUND;
    if (!*pb || !http_base_proto || !c->http_persistent) {
        err = s->io_open(s, pb, filename, AVIO_FLAG_WRITE, options);
#if CONFIG_HTTP_PROTOCOL
    } else {
