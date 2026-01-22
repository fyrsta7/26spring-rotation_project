
    return s->ac3dsp.compute_mantissa_size(mant_cnt);
}


/**
 * Run the bit allocation with a given SNR offset.
 * This calculates the bit allocation pointers that will be used to determine
 * the quantization of each mantissa.
 * @return the number of bits needed for mantissas if the given SNR offset is
 *         is used.
 */
static int bit_alloc(AC3EncodeContext *s, int snr_offset)
{
    int blk, ch;

    snr_offset = (snr_offset - 240) << 2;

    reset_block_bap(s);
    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {
        AC3Block *block = &s->blocks[blk];

        for (ch = !block->cpl_in_use; ch <= s->channels; ch++) {
            /* Currently the only bit allocation parameters which vary across
               blocks within a frame are the exponent values.  We can take
               advantage of that by reusing the bit allocation pointers
               whenever we reuse exponents. */
            if (s->exp_strategy[ch][blk] != EXP_REUSE) {
                s->ac3dsp.bit_alloc_calc_bap(block->mask[ch], block->psd[ch],
                                             s->start_freq[ch], block->end_freq[ch],
                                             snr_offset, s->bit_alloc.floor,
                                             ff_ac3_bap_tab, s->ref_bap[ch][blk]);
            }
        }
    }
    return count_mantissa_bits(s);
}


/**
 * Constant bitrate bit allocation search.
 * Find the largest SNR offset that will allow data to fit in the frame.
 */
static int cbr_bit_allocation(AC3EncodeContext *s)
{
    int ch;
    int bits_left;
    int snr_offset, snr_incr;

    bits_left = 8 * s->frame_size - (s->frame_bits + s->exponent_bits);
    if (bits_left < 0)
        return AVERROR(EINVAL);

    snr_offset = s->coarse_snr_offset << 4;

    /* if previous frame SNR offset was 1023, check if current frame can also
       use SNR offset of 1023. if so, skip the search. */
    if ((snr_offset | s->fine_snr_offset[1]) == 1023) {
        if (bit_alloc(s, 1023) <= bits_left)
            return 0;
    }

    while (snr_offset >= 0 &&
           bit_alloc(s, snr_offset) > bits_left) {
        snr_offset -= 64;
    }
    if (snr_offset < 0)
        return AVERROR(EINVAL);

    FFSWAP(uint8_t *, s->bap_buffer, s->bap1_buffer);
    for (snr_incr = 64; snr_incr > 0; snr_incr >>= 2) {
        while (snr_offset + snr_incr <= 1023 &&
               bit_alloc(s, snr_offset + snr_incr) <= bits_left) {
            snr_offset += snr_incr;
            FFSWAP(uint8_t *, s->bap_buffer, s->bap1_buffer);
        }
    }
    FFSWAP(uint8_t *, s->bap_buffer, s->bap1_buffer);
    reset_block_bap(s);

    s->coarse_snr_offset = snr_offset >> 4;
    for (ch = !s->cpl_on; ch <= s->channels; ch++)
        s->fine_snr_offset[ch] = snr_offset & 0xF;

    return 0;
}

