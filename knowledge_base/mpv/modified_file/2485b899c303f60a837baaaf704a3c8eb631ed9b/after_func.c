            break;
        }
    }

    // Set up reading from new range (as well as writing to it).
    for (int n = 0; n < in->num_streams; n++) {
        struct demux_stream *ds = in->streams[n]->ds;

        ds->queue = range->streams[n];
        ds->refreshing = false;
        ds->eof = false;
    }

    // No point in keeping any junk (especially if old current_range is empty).
    free_empty_cached_ranges(in);
}

static struct demux_packet *find_seek_target(struct demux_queue *queue,
                                             double pts, int flags)
{
    struct demux_packet *target = NULL;
    double target_diff = MP_NOPTS_VALUE;
    for (struct demux_packet *dp = queue->head; dp; dp = dp->next) {
        double range_pts = dp->kf_seek_pts;
        if (!dp->keyframe || range_pts == MP_NOPTS_VALUE)
            continue;

        double diff = range_pts - pts;
        if (flags & SEEK_FORWARD) {
            diff = -diff;
            if (diff > 0)
