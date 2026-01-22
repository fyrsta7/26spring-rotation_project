
/**
 * Count the bits used to encode the frame, minus exponents and mantissas.
 * Bits based on fixed parameters have already been counted, so now we just
 * have to add the bits based on parameters that change during encoding.
 */
static void count_frame_bits(AC3EncodeContext *s)
{
    int blk, ch;
    int frame_bits = 0;

    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {
        uint8_t *exp_strategy = s->blocks[blk].exp_strategy;
        for (ch = 0; ch < s->fbw_channels; ch++) {
            if (exp_strategy[ch] != EXP_REUSE)
                frame_bits += 6 + 2; /* chbwcod[6], gainrng[2] */
        }
    }
    s->frame_bits = s->frame_bits_fixed + frame_bits;
}


/**
 * Calculate the number of bits needed to encode a set of mantissas.
 */
static int compute_mantissa_size(int mant_cnt[5], uint8_t *bap, int nb_coefs)
{
    int bits, b, i;

    bits = 0;
    for (i = 0; i < nb_coefs; i++) {
        b = bap[i];
        if (b <= 4) {
            // bap=1 to bap=4 will be counted in compute_mantissa_size_final
