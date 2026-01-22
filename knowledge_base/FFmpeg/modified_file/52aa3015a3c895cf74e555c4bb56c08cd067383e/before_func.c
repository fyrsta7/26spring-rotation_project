        if (temp1 >= temp2) {
            comp_ppf_gains(fwd_lag, ppf, cur_rate, energy[0], energy[1],
                           energy[2]);
        } else {
            comp_ppf_gains(-back_lag, ppf, cur_rate, energy[0], energy[3],
                           energy[4]);
        }
    }
}

/**
 * Classify frames as voiced/unvoiced.
 *
 * @param p         the context
 * @param pitch_lag decoded pitch_lag
 * @param exc_eng   excitation energy estimation
 * @param scale     scaling factor of exc_eng
 *
 * @return residual interpolation index if voiced, 0 otherwise
 */
static int comp_interp_index(G723_1_Context *p, int pitch_lag,
                             int *exc_eng, int *scale)
{
    int offset = PITCH_MAX + 2 * SUBFRAME_LEN;
    int16_t *buf = p->excitation + offset;

    int index, ccr, tgt_eng, best_eng, temp;

    *scale = scale_vector(p->excitation, FRAME_LEN + PITCH_MAX);

    /* Compute maximum backward cross-correlation */
    ccr   = 0;
    index = autocorr_max(p, offset, &ccr, pitch_lag, SUBFRAME_LEN * 2, -1);
    ccr   = av_clipl_int32((int64_t)ccr + (1 << 15)) >> 16;

    /* Compute target energy */
    tgt_eng  = dot_product(buf, buf, SUBFRAME_LEN * 2, 1);
    *exc_eng = av_clipl_int32((int64_t)tgt_eng + (1 << 15)) >> 16;

    if (ccr <= 0)
        return 0;

    /* Compute best energy */
    best_eng = dot_product(buf - index, buf - index,
                           SUBFRAME_LEN * 2, 1);
    best_eng = av_clipl_int32((int64_t)best_eng + (1 << 15)) >> 16;

    temp = best_eng * *exc_eng >> 3;

    if (temp < ccr * ccr)
        return index;
    else
        return 0;
}

/**
 * Peform residual interpolation based on frame classification.
 *
 * @param buf   decoded excitation vector
 * @param out   output vector
 * @param lag   decoded pitch lag
 * @param gain  interpolated gain
 * @param rseed seed for random number generator
 */
static void residual_interp(int16_t *buf, int16_t *out, int lag,
                            int gain, int *rseed)
{
    int i;
    if (lag) { /* Voiced */
        int16_t *vector_ptr = buf + PITCH_MAX;
