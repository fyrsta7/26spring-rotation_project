 *             may be prefixed with "mp4:")
 */
static int rtmp_open(URLContext *s, const char *uri, int flags)
{
    RTMPContext *rt;
    char proto[8], hostname[256], path[1024], *fname;
    uint8_t buf[2048];
    int port;
    int ret;

    rt = av_mallocz(sizeof(RTMPContext));
    if (!rt)
        return AVERROR(ENOMEM);
    s->priv_data = rt;
    rt->is_input = !(flags & URL_WRONLY);

    ff_url_split(proto, sizeof(proto), NULL, 0, hostname, sizeof(hostname), &port,
                 path, sizeof(path), s->filename);

    if (port < 0)
        port = RTMP_DEFAULT_PORT;
    ff_url_join(buf, sizeof(buf), "tcp", NULL, hostname, port, NULL);

    if (url_open(&rt->stream, buf, URL_RDWR) < 0) {
        av_log(LOG_CONTEXT, AV_LOG_ERROR, "Cannot open connection %s\n", buf);
