static av_cold int init(AVFilterContext *ctx)
{
    IDETContext *idet = ctx->priv;

    idet->eof = 0;
    idet->last_type = UNDETERMINED;
    memset(idet->history, UNDETERMINED, HIST_SIZE);

    if( idet->half_life > 0 )
        idet->decay_coefficient = (uint64_t) round( PRECISION * exp2(-1.0 / idet->half_life) );
    else
        idet->decay_coefficient = PRECISION;

    idet->filter_line = ff_idet_filter_line_c;

    if (ARCH_X86)
        ff_idet_init_x86(idet, 0);

    return 0;
}