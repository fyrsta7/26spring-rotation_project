static void
ngx_http_v2_run_request_handler(ngx_event_t *ev)
{
    ngx_connection_t    *fc;
    ngx_http_request_t  *r;

    fc = ev->data;
    r = fc->data;

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, fc->log, 0,
                   "http2 run request handler");

    ngx_http_v2_run_request(r);
}


ngx_int_t
ngx_http_v2_read_request_body(ngx_http_request_t *r)
{
    off_t                      len;
    size_t                     size;
    ngx_buf_t                 *buf;
    ngx_int_t                  rc;
    ngx_http_v2_stream_t      *stream;
    ngx_http_v2_srv_conf_t    *h2scf;
    ngx_http_request_body_t   *rb;
    ngx_http_core_loc_conf_t  *clcf;
    ngx_http_v2_connection_t  *h2c;

    stream = r->stream;
    rb = r->request_body;

    if (stream->skip_data) {
        r->request_body_no_buffering = 0;
        rb->post_handler(r);
        return NGX_OK;
    }

    h2scf = ngx_http_get_module_srv_conf(r, ngx_http_v2_module);
    clcf = ngx_http_get_module_loc_conf(r, ngx_http_core_module);

    len = r->headers_in.content_length_n;

    if (len < 0 || len > (off_t) clcf->client_body_buffer_size) {
        len = clcf->client_body_buffer_size;

    } else {
        len++;
    }

    if (r->request_body_no_buffering && !stream->in_closed) {

        /*
         * We need a room to store data up to the stream's initial window size,
         * at least until this window will be exhausted.
         */

        if (len < (off_t) h2scf->preread_size) {
            len = h2scf->preread_size;
        }

        if (len > NGX_HTTP_V2_MAX_WINDOW) {
            len = NGX_HTTP_V2_MAX_WINDOW;
        }
    }

    rb->buf = ngx_create_temp_buf(r->pool, (size_t) len);

    if (rb->buf == NULL) {
        stream->skip_data = 1;
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    rb->rest = 1;

    buf = stream->preread;

    if (stream->in_closed) {
        r->request_body_no_buffering = 0;

        if (buf) {
            rc = ngx_http_v2_process_request_body(r, buf->pos,
                                                  buf->last - buf->pos, 1);
            ngx_pfree(r->pool, buf->start);
            return rc;
        }

        return ngx_http_v2_process_request_body(r, NULL, 0, 1);
    }

    if (buf) {
        rc = ngx_http_v2_process_request_body(r, buf->pos,
                                              buf->last - buf->pos, 0);

        ngx_pfree(r->pool, buf->start);

        if (rc != NGX_OK) {
            stream->skip_data = 1;
            return rc;
        }
    }

    if (r->request_body_no_buffering) {
        size = (size_t) len - h2scf->preread_size;

    } else {
        stream->no_flow_control = 1;
        size = NGX_HTTP_V2_MAX_WINDOW - stream->recv_window;
    }

    if (size) {
        if (ngx_http_v2_send_window_update(stream->connection,
                                           stream->node->id, size)
            == NGX_ERROR)
        {
            stream->skip_data = 1;
            return NGX_HTTP_INTERNAL_SERVER_ERROR;
        }

        h2c = stream->connection;

