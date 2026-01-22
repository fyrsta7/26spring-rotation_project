
    if (hdr.bitstream_id <= 10) {
        s->eac3                  = 0;
        s->snr_offset_strategy   = 2;
        s->block_switch_syntax   = 1;
        s->dither_flag_syntax    = 1;
        s->bit_allocation_syntax = 1;
        s->fast_gain_syntax      = 0;
        s->first_cpl_leak        = 0;
        s->dba_syntax            = 1;
        s->skip_syntax           = 1;
        memset(s->channel_uses_aht, 0, sizeof(s->channel_uses_aht));
        return ac3_parse_header(s);
    } else {
        s->eac3 = 1;
        return ff_eac3_parse_header(s);
    }
}

/**
 * Set stereo downmixing coefficients based on frame header info.
 * reference: Section 7.8.2 Downmixing Into Two Channels
 */
static void set_downmix_coeffs(AC3DecodeContext *s)
{
    int i;
    float cmix = gain_levels[center_levels[s->center_mix_level]];
    float smix = gain_levels[surround_levels[s->surround_mix_level]];
