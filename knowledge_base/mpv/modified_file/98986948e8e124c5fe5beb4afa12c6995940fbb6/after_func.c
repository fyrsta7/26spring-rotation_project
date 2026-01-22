};

static const char *get_lavc_format(const char *format)
{
    // For the hack involving parse_webvtt().
    if (format && strcmp(format, "webvtt-webm") == 0)
        format = "webvtt";
    // Most text subtitles are srt/html style anyway.
    if (format && strcmp(format, "text") == 0)
        format = "subrip";
    return format;
