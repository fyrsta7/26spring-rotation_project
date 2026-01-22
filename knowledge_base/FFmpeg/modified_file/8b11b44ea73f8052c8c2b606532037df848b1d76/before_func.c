        int nf = s->channel_mode - 2;
        s->downmix_coeffs[nf][0] = s->downmix_coeffs[nf][1] = smix * LEVEL_MINUS_3DB;
    }
    if(s->channel_mode == AC3_CHMODE_2F2R || s->channel_mode == AC3_CHMODE_3F2R) {
        int nf = s->channel_mode - 4;
        s->downmix_coeffs[nf][0] = s->downmix_coeffs[nf+1][1] = smix;
    }

    /* renormalize */
    norm0 = norm1 = 0.0;
    for(i=0; i<s->fbw_channels; i++) {
        norm0 += s->downmix_coeffs[i][0];
        norm1 += s->downmix_coeffs[i][1];
    }
    norm0 = 1.0f / norm0;
    norm1 = 1.0f / norm1;
    for(i=0; i<s->fbw_channels; i++) {
        s->downmix_coeffs[i][0] *= norm0;
        s->downmix_coeffs[i][1] *= norm1;
