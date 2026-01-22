        if (qc->min_rtt + ack_delay < latest_rtt) {
            adjusted_rtt -= ack_delay;
        }

        qc->avg_rtt = 0.875 * qc->avg_rtt + 0.125 * adjusted_rtt;
        rttvar_sample = ngx_abs((ngx_msec_int_t) (qc->avg_rtt - adjusted_rtt));
        qc->rttvar = 0.75 * qc->rttvar + 0.25 * rttvar_sample;
    }

    ngx_log_debug4(NGX_LOG_DEBUG_EVENT, c->log, 0,
                   "quic rtt sample latest:%M min:%M avg:%M var:%M",
                   latest_rtt, qc->min_rtt, qc->avg_rtt, qc->rttvar);
}


static ngx_int_t
ngx_quic_handle_ack_frame_range(ngx_connection_t *c, ngx_quic_send_ctx_t *ctx,
    uint64_t min, uint64_t max, ngx_quic_ack_stat_t *st)
{
    ngx_uint_t              found;
    ngx_queue_t            *q;
    ngx_quic_frame_t       *f;
    ngx_quic_connection_t  *qc;

    qc = ngx_quic_get_connection(c);

    st->max_pn = NGX_TIMER_INFINITE;
    found = 0;

    q = ngx_queue_last(&ctx->sent);

    while (q != ngx_queue_sentinel(&ctx->sent)) {

        f = ngx_queue_data(q, ngx_quic_frame_t, queue);
        q = ngx_queue_prev(q);

        if (f->pnum >= min && f->pnum <= max) {
            ngx_quic_congestion_ack(c, f);

            switch (f->type) {
            case NGX_QUIC_FT_ACK:
            case NGX_QUIC_FT_ACK_ECN:
                ngx_quic_drop_ack_ranges(c, ctx, f->u.ack.largest);
                break;

            case NGX_QUIC_FT_STREAM:
                ngx_quic_handle_stream_ack(c, f);
                break;
            }

            if (f->pnum == max) {
                st->max_pn = f->last;
            }

            /* save earliest and latest send times of frames ack'ed */
            if (st->oldest == NGX_TIMER_INFINITE || f->last < st->oldest) {
                st->oldest = f->last;
            }

            if (st->newest == NGX_TIMER_INFINITE || f->last > st->newest) {
                st->newest = f->last;
            }

            ngx_queue_remove(&f->queue);
            ngx_quic_free_frame(c, f);
            found = 1;
        }
    }

    if (!found) {

        if (max < ctx->pnum) {
            /* duplicate ACK or ACK for non-ack-eliciting frame */
            return NGX_OK;
        }

        ngx_log_error(NGX_LOG_INFO, c->log, 0,
                      "quic ACK for the packet not sent");

