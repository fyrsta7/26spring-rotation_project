{
    int qscale;
    if (s->mpeg2) {
        if (s->q_scale_type) {
            qscale = non_linear_qscale[get_bits(&s->gb, 5)];
        } else {
            qscale = get_bits(&s->gb, 5) << 1;
        }
    } else {
        /* for mpeg1, we use the generic unquant code */
        qscale = get_bits(&s->gb, 5);
    }
    return qscale;
}

/* motion type (for mpeg2) */
#define MT_FIELD 1
#define MT_FRAME 2
#define MT_16X8  2
#define MT_DMV   3

static int mpeg_decode_mb(MpegEncContext *s,
                          DCTELEM block[6][64])
{
    int i, j, k, cbp, val, code, mb_type, motion_type;
    
    /* skip mb handling */
    if (s->mb_incr == 0) {
        /* read again increment */
        s->mb_incr = 1;
        for(;;) {
            code = get_vlc(&s->gb, &mbincr_vlc);
            if (code < 0)
                return 1; /* error = end of slice */
            if (code >= 33) {
                if (code == 33) {
                    s->mb_incr += 33;
                }
                /* otherwise, stuffing, nothing to do */
            } else {
                s->mb_incr += code;
                break;
            }
        }
    }
    if (++s->mb_x >= s->mb_width) {
        s->mb_x = 0;
        if (s->mb_y >= (s->mb_height - 1))
            return -1;
        s->mb_y++;
    }
    dprintf("decode_mb: x=%d y=%d\n", s->mb_x, s->mb_y);

    if (--s->mb_incr != 0) {
        /* skip mb */
        s->mb_intra = 0;
        for(i=0;i<6;i++)
            s->block_last_index[i] = -1;
        s->mv_type = MV_TYPE_16X16;
        if (s->pict_type == P_TYPE) {
            /* if P type, zero motion vector is implied */
            s->mv_dir = MV_DIR_FORWARD;
            s->mv[0][0][0] = s->mv[0][0][1] = 0;
            s->last_mv[0][0][0] = s->last_mv[0][0][1] = 0;
            s->last_mv[0][1][0] = s->last_mv[0][1][1] = 0;
        } else {
            /* if B type, reuse previous vectors and directions */
            s->mv[0][0][0] = s->last_mv[0][0][0];
            s->mv[0][0][1] = s->last_mv[0][0][1];
            s->mv[1][0][0] = s->last_mv[1][0][0];
            s->mv[1][0][1] = s->last_mv[1][0][1];
        }
        s->mb_skiped = 1;
        return 0;
    }

    switch(s->pict_type) {
    default:
    case I_TYPE:
        if (get_bits1(&s->gb) == 0) {
            if (get_bits1(&s->gb) == 0)
                return -1;
            mb_type = MB_QUANT | MB_INTRA;
        } else {
            mb_type = MB_INTRA;
        }
        break;
    case P_TYPE:
        mb_type = get_vlc(&s->gb, &mb_ptype_vlc);
        if (mb_type < 0)
            return -1;
        break;
    case B_TYPE:
        mb_type = get_vlc(&s->gb, &mb_btype_vlc);
        if (mb_type < 0)
            return -1;
        break;
    }
    dprintf("mb_type=%x\n", mb_type);
    motion_type = 0; /* avoid warning */
    if (mb_type & (MB_FOR|MB_BACK)) {
        /* get additionnal motion vector type */
        if (s->picture_structure == PICT_FRAME && s->frame_pred_frame_dct) 
            motion_type = MT_FRAME;
        else
            motion_type = get_bits(&s->gb, 2);
    }
    /* compute dct type */
    if (s->picture_structure == PICT_FRAME && 
        !s->frame_pred_frame_dct &&
        (mb_type & (MB_PAT | MB_INTRA))) {
        s->interlaced_dct = get_bits1(&s->gb);
#ifdef DEBUG
        if (s->interlaced_dct)
            printf("interlaced_dct\n");
#endif
    } else {
        s->interlaced_dct = 0; /* frame based */
    }

    if (mb_type & MB_QUANT) {
        s->qscale = get_qscale(s);
    }
    if (mb_type & MB_INTRA) {
        if (s->concealment_motion_vectors) {
            /* just parse them */
            if (s->picture_structure != PICT_FRAME) 
                skip_bits1(&s->gb); /* field select */
            mpeg_decode_motion(s, s->mpeg_f_code[0][0], 0);
            mpeg_decode_motion(s, s->mpeg_f_code[0][1], 0);
        }
        s->mb_intra = 1;
        cbp = 0x3f;
        memset(s->last_mv, 0, sizeof(s->last_mv)); /* reset mv prediction */
    } else {
        s->mb_intra = 0;
        cbp = 0;
    }
    /* special case of implicit zero motion vector */
    if (s->pict_type == P_TYPE && !(mb_type & MB_FOR)) {
        s->mv_dir = MV_DIR_FORWARD;
        s->mv_type = MV_TYPE_16X16;
        s->last_mv[0][0][0] = 0;
        s->last_mv[0][0][1] = 0;
        s->last_mv[0][1][0] = 0;
        s->last_mv[0][1][1] = 0;
        s->mv[0][0][0] = 0;
        s->mv[0][0][1] = 0;
    } else if (mb_type & (MB_FOR | MB_BACK)) {
        /* motion vectors */
        s->mv_dir = 0;
        for(i=0;i<2;i++) {
            if (mb_type & (MB_FOR >> i)) {
                s->mv_dir |= (MV_DIR_FORWARD >> i);
                dprintf("motion_type=%d\n", motion_type);
                switch(motion_type) {
                case MT_FRAME: /* or MT_16X8 */
                    if (s->picture_structure == PICT_FRAME) {
                        /* MT_FRAME */
                        s->mv_type = MV_TYPE_16X16;
                        for(k=0;k<2;k++) {
                            val = mpeg_decode_motion(s, s->mpeg_f_code[i][k], 
                                                     s->last_mv[i][0][k]);
                            s->last_mv[i][0][k] = val;
                            s->last_mv[i][1][k] = val;
                            /* full_pel: only for mpeg1 */
                            if (s->full_pel[i])
                                val = val << 1;
                            s->mv[i][0][k] = val;
                            dprintf("mv%d: %d\n", k, val);
                        }
                    } else {
                        /* MT_16X8 */
                        s->mv_type = MV_TYPE_16X8;
                        for(j=0;j<2;j++) {
                            s->field_select[i][j] = get_bits1(&s->gb);
                            for(k=0;k<2;k++) {
                                val = mpeg_decode_motion(s, s->mpeg_f_code[i][k],
                                                         s->last_mv[i][j][k]);
                                s->last_mv[i][j][k] = val;
                                s->mv[i][j][k] = val;
                            }
                        }
                    }
                    break;
                case MT_FIELD:
                    if (s->picture_structure == PICT_FRAME) {
                        s->mv_type = MV_TYPE_FIELD;
                        for(j=0;j<2;j++) {
                            s->field_select[i][j] = get_bits1(&s->gb);
                            val = mpeg_decode_motion(s, s->mpeg_f_code[i][0],
                                                     s->last_mv[i][j][0]);
                            s->last_mv[i][j][0] = val;
                            s->mv[i][j][0] = val;
                            dprintf("fmx=%d\n", val);
                            val = mpeg_decode_motion(s, s->mpeg_f_code[i][1],
                                                     s->last_mv[i][j][1] >> 1);
                            s->last_mv[i][j][1] = val << 1;
                            s->mv[i][j][1] = val;
                            dprintf("fmy=%d\n", val);
                        }
                    } else {
                        s->mv_type = MV_TYPE_16X16;
                        s->field_select[i][0] = get_bits1(&s->gb);
                        for(k=0;k<2;k++) {
                            val = mpeg_decode_motion(s, s->mpeg_f_code[i][k],
                                                     s->last_mv[i][0][k]);
                            s->last_mv[i][0][k] = val;
                            s->last_mv[i][1][k] = val;
                            s->mv[i][0][k] = val;
                        }
                    }
                    break;
                case MT_DMV:
                    {
                        int dmx, dmy, mx, my, m;

                        mx = mpeg_decode_motion(s, s->mpeg_f_code[i][0], 
                                                s->last_mv[i][0][0]);
                        s->last_mv[i][0][0] = mx;
                        s->last_mv[i][1][0] = mx;
                        dmx = get_dmv(s);
                        my = mpeg_decode_motion(s, s->mpeg_f_code[i][1], 
                                                s->last_mv[i][0][1] >> 1);
                        dmy = get_dmv(s);
                        s->mv_type = MV_TYPE_DMV;
                        /* XXX: totally broken */
                        if (s->picture_structure == PICT_FRAME) {
                            s->last_mv[i][0][1] = my << 1;
                            s->last_mv[i][1][1] = my << 1;

                            m = s->top_field_first ? 1 : 3;
                            /* top -> top pred */
                            s->mv[i][0][0] = mx; 
                            s->mv[i][0][1] = my << 1;
                            s->mv[i][1][0] = ((mx * m + (mx > 0)) >> 1) + dmx;
                            s->mv[i][1][1] = ((my * m + (my > 0)) >> 1) + dmy - 1;
                            m = 4 - m;
                            s->mv[i][2][0] = mx;
                            s->mv[i][2][1] = my << 1;
                            s->mv[i][3][0] = ((mx * m + (mx > 0)) >> 1) + dmx;
                            s->mv[i][3][1] = ((my * m + (my > 0)) >> 1) + dmy + 1;
                        } else {
                            s->last_mv[i][0][1] = my;
                            s->last_mv[i][1][1] = my;
                            s->mv[i][0][0] = mx;
                            s->mv[i][0][1] = my;
                            s->mv[i][1][0] = ((mx + (mx > 0)) >> 1) + dmx;
                            s->mv[i][1][1] = ((my + (my > 0)) >> 1) + dmy - 1 
                                /* + 2 * cur_field */;
                        }
                    }
                    break;
                }
            }
        }
    }

    if ((mb_type & MB_INTRA) && s->concealment_motion_vectors) {
        skip_bits1(&s->gb); /* marker */
    }
    
    if (mb_type & MB_PAT) {
        cbp = get_vlc(&s->gb, &mb_pat_vlc);
        if (cbp < 0)
            return -1;
        cbp++;
    }
    dprintf("cbp=%x\n", cbp);

    if (s->mpeg2) {
        if (s->mb_intra) {
            for(i=0;i<6;i++) {
                if (cbp & (1 << (5 - i))) {
                    if (mpeg2_decode_block_intra(s, block[i], i) < 0)
                        return -1;
                } else {
                    s->block_last_index[i] = -1;
                }
            }
        } else {
