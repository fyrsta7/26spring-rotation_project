
    gap = smallest - 2 - pn;
    range = 0;

insert:

    if (ctx->nranges < NGX_QUIC_MAX_RANGES) {
        ctx->nranges++;
    }

    ngx_memmove(&ctx->ranges[i + 1], &ctx->ranges[i],
                sizeof(ngx_quic_ack_range_t) * (ctx->nranges - i - 1));

    ctx->ranges[i].gap = gap;
    ctx->ranges[i].range = range;

    return NGX_OK;
}


ngx_int_t
ngx_quic_generate_ack(ngx_connection_t *c, ngx_quic_send_ctx_t *ctx)
{
    ngx_msec_t              delay;
    ngx_quic_connection_t  *qc;

    if (!ctx->send_ack) {
        return NGX_OK;
    }

    if (ctx->level == ssl_encryption_application) {

        delay = ngx_current_msec - ctx->ack_delay_start;
        qc = ngx_quic_get_connection(c);

