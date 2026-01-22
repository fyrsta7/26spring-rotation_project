                put_bits(&s->pb, table_vlc[code][1] + 1,
                         (table_vlc[code][0] << 1) + sign);
            } else {
                /* escape seems to be pretty rare <5% so I do not optimize it */
                put_bits(&s->pb, table_vlc[111][1], table_vlc[111][0]);
                /* escape: only clip in this case */
                put_bits(&s->pb, 6, run);
                if (s->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
                    if (alevel < 128) {
                        put_sbits(&s->pb, 8, level);
                    } else {
                        if (level < 0)
                            put_bits(&s->pb, 16, 0x8001 + level + 255);
                        else
                            put_sbits(&s->pb, 16, level);
                    }
                } else {
                    put_sbits(&s->pb, 12, level);
                }
            }
            last_non_zero = i;
        }
    }
    /* end of block */
    put_bits(&s->pb, table_vlc[112][1], table_vlc[112][0]);
}

static av_always_inline void mpeg1_encode_mb_internal(MpegEncContext *s,
                                                      int16_t block[8][64],
                                                      int motion_x, int motion_y,
                                                      int mb_block_count,
                                                      int chroma_y_shift)
{
    int i, cbp;
    const int mb_x     = s->mb_x;
    const int mb_y     = s->mb_y;
    const int first_mb = mb_x == s->resync_mb_x && mb_y == s->resync_mb_y;

    /* compute cbp */
    cbp = 0;
    for (i = 0; i < mb_block_count; i++)
        if (s->block_last_index[i] >= 0)
            cbp |= 1 << (mb_block_count - 1 - i);

    if (cbp == 0 && !first_mb && s->mv_type == MV_TYPE_16X16 &&
        (mb_x != s->mb_width - 1 ||
         (mb_y != s->end_mb_y - 1 && s->codec_id == AV_CODEC_ID_MPEG1VIDEO)) &&
        ((s->pict_type == AV_PICTURE_TYPE_P && (motion_x | motion_y) == 0) ||
         (s->pict_type == AV_PICTURE_TYPE_B && s->mv_dir == s->last_mv_dir &&
          (((s->mv_dir & MV_DIR_FORWARD)
            ? ((s->mv[0][0][0] - s->last_mv[0][0][0]) |
               (s->mv[0][0][1] - s->last_mv[0][0][1])) : 0) |
           ((s->mv_dir & MV_DIR_BACKWARD)
            ? ((s->mv[1][0][0] - s->last_mv[1][0][0]) |
               (s->mv[1][0][1] - s->last_mv[1][0][1])) : 0)) == 0))) {
        s->mb_skip_run++;
        s->qscale -= s->dquant;
        s->skip_count++;
        s->misc_bits++;
        s->last_bits++;
        if (s->pict_type == AV_PICTURE_TYPE_P) {
            s->last_mv[0][0][0] =
            s->last_mv[0][0][1] =
            s->last_mv[0][1][0] =
            s->last_mv[0][1][1] = 0;
        }
    } else {
        if (first_mb) {
            av_assert0(s->mb_skip_run == 0);
            encode_mb_skip_run(s, s->mb_x);
        } else {
            encode_mb_skip_run(s, s->mb_skip_run);
        }

        if (s->pict_type == AV_PICTURE_TYPE_I) {
            if (s->dquant && cbp) {
                /* macroblock_type: macroblock_quant = 1 */
                put_mb_modes(s, 2, 1, 0, 0);
                put_qscale(s);
            } else {
                /* macroblock_type: macroblock_quant = 0 */
                put_mb_modes(s, 1, 1, 0, 0);
                s->qscale -= s->dquant;
            }
            s->misc_bits += get_bits_diff(s);
            s->i_count++;
        } else if (s->mb_intra) {
            if (s->dquant && cbp) {
                put_mb_modes(s, 6, 0x01, 0, 0);
                put_qscale(s);
            } else {
                put_mb_modes(s, 5, 0x03, 0, 0);
                s->qscale -= s->dquant;
            }
            s->misc_bits += get_bits_diff(s);
            s->i_count++;
            memset(s->last_mv, 0, sizeof(s->last_mv));
        } else if (s->pict_type == AV_PICTURE_TYPE_P) {
            if (s->mv_type == MV_TYPE_16X16) {
                if (cbp != 0) {
                    if ((motion_x | motion_y) == 0) {
                        if (s->dquant) {
                            /* macroblock_pattern & quant */
                            put_mb_modes(s, 5, 1, 0, 0);
                            put_qscale(s);
                        } else {
                            /* macroblock_pattern only */
                            put_mb_modes(s, 2, 1, 0, 0);
                        }
                        s->misc_bits += get_bits_diff(s);
                    } else {
                        if (s->dquant) {
                            put_mb_modes(s, 5, 2, 1, 0);    /* motion + cbp */
                            put_qscale(s);
                        } else {
                            put_mb_modes(s, 1, 1, 1, 0);    /* motion + cbp */
                        }
                        s->misc_bits += get_bits_diff(s);
                        // RAL: f_code parameter added
                        mpeg1_encode_motion(s,
                                            motion_x - s->last_mv[0][0][0],
                                            s->f_code);
                        // RAL: f_code parameter added
                        mpeg1_encode_motion(s,
                                            motion_y - s->last_mv[0][0][1],
                                            s->f_code);
                        s->mv_bits += get_bits_diff(s);
                    }
                } else {
                    put_bits(&s->pb, 3, 1);         /* motion only */
                    if (!s->frame_pred_frame_dct)
                        put_bits(&s->pb, 2, 2);     /* motion_type: frame */
                    s->misc_bits += get_bits_diff(s);
                    // RAL: f_code parameter added
                    mpeg1_encode_motion(s,
                                        motion_x - s->last_mv[0][0][0],
                                        s->f_code);
                    // RAL: f_code parameter added
                    mpeg1_encode_motion(s,
                                        motion_y - s->last_mv[0][0][1],
                                        s->f_code);
                    s->qscale  -= s->dquant;
                    s->mv_bits += get_bits_diff(s);
                }
                s->last_mv[0][1][0] = s->last_mv[0][0][0] = motion_x;
                s->last_mv[0][1][1] = s->last_mv[0][0][1] = motion_y;
            } else {
                av_assert2(!s->frame_pred_frame_dct && s->mv_type == MV_TYPE_FIELD);

                if (cbp) {
                    if (s->dquant) {
                        put_mb_modes(s, 5, 2, 1, 1);    /* motion + cbp */
                        put_qscale(s);
                    } else {
                        put_mb_modes(s, 1, 1, 1, 1);    /* motion + cbp */
                    }
                } else {
                    put_bits(&s->pb, 3, 1);             /* motion only */
                    put_bits(&s->pb, 2, 1);             /* motion_type: field */
                    s->qscale -= s->dquant;
                }
                s->misc_bits += get_bits_diff(s);
                for (i = 0; i < 2; i++) {
                    put_bits(&s->pb, 1, s->field_select[0][i]);
                    mpeg1_encode_motion(s,
                                        s->mv[0][i][0] - s->last_mv[0][i][0],
                                        s->f_code);
                    mpeg1_encode_motion(s,
                                        s->mv[0][i][1] - (s->last_mv[0][i][1] >> 1),
                                        s->f_code);
                    s->last_mv[0][i][0] = s->mv[0][i][0];
                    s->last_mv[0][i][1] = 2 * s->mv[0][i][1];
                }
                s->mv_bits += get_bits_diff(s);
            }
            if (cbp) {
                if (chroma_y_shift) {
                    put_bits(&s->pb,
                             ff_mpeg12_mbPatTable[cbp][1],
                             ff_mpeg12_mbPatTable[cbp][0]);
                } else {
                    put_bits(&s->pb,
                             ff_mpeg12_mbPatTable[cbp >> 2][1],
                             ff_mpeg12_mbPatTable[cbp >> 2][0]);
                    put_sbits(&s->pb, 2, cbp);
                }
            }
            s->f_count++;
        } else {
            if (s->mv_type == MV_TYPE_16X16) {
                if (cbp) {                      // With coded bloc pattern
                    if (s->dquant) {
                        if (s->mv_dir == MV_DIR_FORWARD)
                            put_mb_modes(s, 6, 3, 1, 0);
                        else
                            put_mb_modes(s, 8 - s->mv_dir, 2, 1, 0);
                        put_qscale(s);
                    } else {
                        put_mb_modes(s, 5 - s->mv_dir, 3, 1, 0);
                    }
                } else {                        // No coded bloc pattern
                    put_bits(&s->pb, 5 - s->mv_dir, 2);
                    if (!s->frame_pred_frame_dct)
                        put_bits(&s->pb, 2, 2); /* motion_type: frame */
                    s->qscale -= s->dquant;
                }
                s->misc_bits += get_bits_diff(s);
                if (s->mv_dir & MV_DIR_FORWARD) {
                    mpeg1_encode_motion(s,
                                        s->mv[0][0][0] - s->last_mv[0][0][0],
                                        s->f_code);
                    mpeg1_encode_motion(s,
                                        s->mv[0][0][1] - s->last_mv[0][0][1],
                                        s->f_code);
                    s->last_mv[0][0][0] =
                    s->last_mv[0][1][0] = s->mv[0][0][0];
                    s->last_mv[0][0][1] =
                    s->last_mv[0][1][1] = s->mv[0][0][1];
                    s->f_count++;
                }
                if (s->mv_dir & MV_DIR_BACKWARD) {
                    mpeg1_encode_motion(s,
                                        s->mv[1][0][0] - s->last_mv[1][0][0],
                                        s->b_code);
                    mpeg1_encode_motion(s,
                                        s->mv[1][0][1] - s->last_mv[1][0][1],
                                        s->b_code);
                    s->last_mv[1][0][0] =
                    s->last_mv[1][1][0] = s->mv[1][0][0];
                    s->last_mv[1][0][1] =
                    s->last_mv[1][1][1] = s->mv[1][0][1];
                    s->b_count++;
                }
            } else {
                av_assert2(s->mv_type == MV_TYPE_FIELD);
                av_assert2(!s->frame_pred_frame_dct);
                if (cbp) {                      // With coded bloc pattern
                    if (s->dquant) {
                        if (s->mv_dir == MV_DIR_FORWARD)
                            put_mb_modes(s, 6, 3, 1, 1);
                        else
                            put_mb_modes(s, 8 - s->mv_dir, 2, 1, 1);
                        put_qscale(s);
                    } else {
                        put_mb_modes(s, 5 - s->mv_dir, 3, 1, 1);
                    }
                } else {                        // No coded bloc pattern
                    put_bits(&s->pb, 5 - s->mv_dir, 2);
                    put_bits(&s->pb, 2, 1);     /* motion_type: field */
                    s->qscale -= s->dquant;
                }
                s->misc_bits += get_bits_diff(s);
                if (s->mv_dir & MV_DIR_FORWARD) {
                    for (i = 0; i < 2; i++) {
                        put_bits(&s->pb, 1, s->field_select[0][i]);
                        mpeg1_encode_motion(s,
                                            s->mv[0][i][0] - s->last_mv[0][i][0],
                                            s->f_code);
                        mpeg1_encode_motion(s,
                                            s->mv[0][i][1] - (s->last_mv[0][i][1] >> 1),
                                            s->f_code);
                        s->last_mv[0][i][0] = s->mv[0][i][0];
                        s->last_mv[0][i][1] = s->mv[0][i][1] * 2;
                    }
                    s->f_count++;
                }
                if (s->mv_dir & MV_DIR_BACKWARD) {
                    for (i = 0; i < 2; i++) {
                        put_bits(&s->pb, 1, s->field_select[1][i]);
                        mpeg1_encode_motion(s,
                                            s->mv[1][i][0] - s->last_mv[1][i][0],
                                            s->b_code);
                        mpeg1_encode_motion(s,
                                            s->mv[1][i][1] - (s->last_mv[1][i][1] >> 1),
                                            s->b_code);
                        s->last_mv[1][i][0] = s->mv[1][i][0];
                        s->last_mv[1][i][1] = s->mv[1][i][1] * 2;
                    }
