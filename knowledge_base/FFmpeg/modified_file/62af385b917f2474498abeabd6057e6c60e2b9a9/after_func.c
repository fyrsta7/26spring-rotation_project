                      const AVStream *st)
{
    pkt_dump_internal(avcl, NULL, level, pkt, dump_payload, st->time_base);
}


static void print_fps(double d, const char *postfix)
{
    uint64_t v = lrintf(d * 100);
    if (!v)
        av_log(NULL, AV_LOG_INFO, "%1.4f %s", d, postfix);
    else if (v % 100)
        av_log(NULL, AV_LOG_INFO, "%3.2f %s", d, postfix);
    else if (v % (100 * 1000))
        av_log(NULL, AV_LOG_INFO, "%1.0f %s", d, postfix);
    else
        av_log(NULL, AV_LOG_INFO, "%1.0fk %s", d / 1000, postfix);
}

static void dump_metadata(void *ctx, const AVDictionary *m, const char *indent)
{
    if (m && !(av_dict_count(m) == 1 && av_dict_get(m, "language", NULL, 0))) {
        const AVDictionaryEntry *tag = NULL;
