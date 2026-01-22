
done:
    talloc_free(log);
    return s;
}

struct stream *stream_open(const char *filename, struct mpv_global *global)
{
    return stream_create(filename, STREAM_READ, global);
}

stream_t *open_output_stream(const char *filename, struct mpv_global *global)
{
    return stream_create(filename, STREAM_WRITE, global);
}

static int stream_reconnect(stream_t *s)
{
#define MAX_RECONNECT_RETRIES 5
#define RECONNECT_SLEEP_MAX_MS 500
    if (!s->streaming)
        return 0;
    if (!(s->flags & MP_STREAM_SEEK_FW))
        return 0;
    int64_t pos = s->pos;
    int sleep_ms = 5;
    for (int retry = 0; retry < MAX_RECONNECT_RETRIES; retry++) {
        MP_WARN(s, "Connection lost! Attempting to reconnect (%d)...\n", retry + 1);

        if (retry) {
            mp_sleep_us(sleep_ms * 1000);
            sleep_ms = MPMIN(sleep_ms * 2, RECONNECT_SLEEP_MAX_MS);
        }

        if (stream_check_interrupt(0))
            return 0;
