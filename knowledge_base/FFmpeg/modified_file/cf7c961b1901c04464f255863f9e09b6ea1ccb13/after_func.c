     *   SNR offsets do not change between blocks
     *   no delta bit allocation
     *   no skipped data
     *   no auxilliary data
     */

    /* header size */
    frame_bits = 65;
    frame_bits += frame_bits_inc[s->channel_mode];

    /* audio blocks */
    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {
        frame_bits += s->fbw_channels * 2 + 2; /* blksw * c, dithflag * c, dynrnge, cplstre */
        if (s->channel_mode == AC3_CHMODE_STEREO) {
            frame_bits++; /* rematstr */
            if (!blk)
                frame_bits += 4;
        }
        frame_bits += 2 * s->fbw_channels; /* chexpstr[2] * c */
        if (s->lfe_on)
            frame_bits++; /* lfeexpstr */
        frame_bits++; /* baie */
