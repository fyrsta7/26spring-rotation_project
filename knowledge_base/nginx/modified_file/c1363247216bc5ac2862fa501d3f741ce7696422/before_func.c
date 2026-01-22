

static ngx_quic_stream_t *
ngx_quic_get_stream(ngx_connection_t *c, uint64_t id)
{
    uint64_t                min_id;
    ngx_event_t            *rev;
    ngx_quic_stream_t      *qs;
    ngx_quic_connection_t  *qc;

    qc = ngx_quic_get_connection(c);

    qs = ngx_quic_find_stream(&qc->streams.tree, id);

    if (qs) {
        return qs;
    }

    if (qc->shutdown || qc->closing) {
        return NGX_QUIC_STREAM_GONE;
    }

    ngx_log_debug1(NGX_LOG_DEBUG_EVENT, c->log, 0,
                   "quic stream id:0x%xL is missing", id);

    if (id & NGX_QUIC_STREAM_UNIDIRECTIONAL) {

        if (id & NGX_QUIC_STREAM_SERVER_INITIATED) {
            if ((id >> 2) < qc->streams.server_streams_uni) {
                return NGX_QUIC_STREAM_GONE;
            }

            qc->error = NGX_QUIC_ERR_STREAM_STATE_ERROR;
            return NULL;
        }

        if ((id >> 2) < qc->streams.client_streams_uni) {
            return NGX_QUIC_STREAM_GONE;
        }

        if ((id >> 2) >= qc->streams.client_max_streams_uni) {
            qc->error = NGX_QUIC_ERR_STREAM_LIMIT_ERROR;
            return NULL;
        }

        min_id = (qc->streams.client_streams_uni << 2)
                 | NGX_QUIC_STREAM_UNIDIRECTIONAL;
        qc->streams.client_streams_uni = (id >> 2) + 1;

    } else {

        if (id & NGX_QUIC_STREAM_SERVER_INITIATED) {
            if ((id >> 2) < qc->streams.server_streams_bidi) {
                return NGX_QUIC_STREAM_GONE;
            }

            qc->error = NGX_QUIC_ERR_STREAM_STATE_ERROR;
            return NULL;
        }

        if ((id >> 2) < qc->streams.client_streams_bidi) {
            return NGX_QUIC_STREAM_GONE;
        }

        if ((id >> 2) >= qc->streams.client_max_streams_bidi) {
            qc->error = NGX_QUIC_ERR_STREAM_LIMIT_ERROR;
            return NULL;
        }

        min_id = (qc->streams.client_streams_bidi << 2);
        qc->streams.client_streams_bidi = (id >> 2) + 1;
    }

    /*
     * RFC 9000, 2.1.  Stream Types and Identifiers
     *
     * successive streams of each type are created with numerically increasing
     * stream IDs.  A stream ID that is used out of order results in all
     * streams of that type with lower-numbered stream IDs also being opened.
     */

#if (NGX_SUPPRESS_WARN)
    qs = NULL;
#endif

    for ( /* void */ ; min_id <= id; min_id += 0x04) {

        qs = ngx_quic_create_stream(c, min_id);

        if (qs == NULL) {
            if (ngx_quic_reject_stream(c, min_id) != NGX_OK) {
                return NULL;
            }

            continue;
        }

        ngx_queue_insert_tail(&qc->streams.uninitialized, &qs->queue);

        rev = qs->connection->read;
        rev->handler = ngx_quic_init_stream_handler;

        if (qc->streams.initialized) {
            ngx_post_event(rev, &ngx_posted_events);
        }
