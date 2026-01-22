
    struct stream *stream = demuxer->stream;
    if (stream) {
        new += stream->total_unbuffered_read_bytes;
        stream->total_unbuffered_read_bytes = 0;
        new_seeks += stream->total_stream_seeks;
        stream->total_stream_seeks = 0;
    }

    in->cache_unbuffered_read_bytes += new;
    in->hack_unbuffered_read_bytes += new;
    in->byte_level_seeks += new_seeks;
}

// must be called locked, temporarily unlocks
static void update_cache(struct demux_internal *in)
{
    struct demuxer *demuxer = in->d_thread;
    struct stream *stream = demuxer->stream;

    struct mp_tags *stream_metadata = NULL;

    // Don't lock while querying the stream.
    pthread_mutex_unlock(&in->lock);

    int64_t stream_size = -1;
    if (stream) {
        stream_size = stream_get_size(stream);
        stream_control(stream, STREAM_CTRL_GET_METADATA, &stream_metadata);
    }

    update_bytes_read(in);

    pthread_mutex_lock(&in->lock);

    in->stream_size = stream_size;
    if (stream_metadata) {
        add_timed_metadata(in, stream_metadata, NULL, MP_NOPTS_VALUE);
        talloc_free(stream_metadata);
    }
