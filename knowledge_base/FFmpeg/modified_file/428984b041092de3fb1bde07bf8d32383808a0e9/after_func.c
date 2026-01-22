                for (k = 1; k < 6; k++) {
                    motion_x[k] = motion_x[0];
                    motion_y[k] = motion_y[0];
                }

                /* vector maintenance */
                prior_last_motion_x = last_motion_x;
                prior_last_motion_y = last_motion_y;
                last_motion_x = motion_x[0];
                last_motion_y = motion_y[0];
                break;

            default:
                /* covers intra, inter without MV, golden without MV */
                memset(motion_x, 0, 6 * sizeof(int));
                memset(motion_y, 0, 6 * sizeof(int));

                /* no vector maintenance */
                break;
            }

            /* assign the motion vectors to the correct fragments */
            for (k = 0; k < 6; k++) {
                current_fragment =
                    s->macroblock_fragments[current_macroblock * 6 + k];
                if (current_fragment == -1)
                    continue;
                if (current_fragment >= s->fragment_count) {
                    av_log(s->avctx, AV_LOG_ERROR, "  vp3:unpack_vectors(): bad fragment number (%d >= %d)\n",
                        current_fragment, s->fragment_count);
                    return 1;
                }
                s->all_fragments[current_fragment].motion_x = motion_x[k];
                s->all_fragments[current_fragment].motion_y = motion_y[k];
            }
        }
    }

    return 0;
}

static int unpack_block_qpis(Vp3DecodeContext *s, GetBitContext *gb)
{
    int qpi, i, j, bit, run_length, blocks_decoded, num_blocks_at_qpi;
    int num_blocks = s->coded_fragment_list_index;

    for (qpi = 0; qpi < s->nqps-1 && num_blocks > 0; qpi++) {
        i = blocks_decoded = num_blocks_at_qpi = 0;

        bit = get_bits1(gb);

        do {
            run_length = get_vlc2(gb, s->superblock_run_length_vlc.table, 6, 2) + 1;
            if (run_length == 34)
                run_length += get_bits(gb, 12);
            blocks_decoded += run_length;

            if (!bit)
                num_blocks_at_qpi += run_length;

            for (j = 0; j < run_length; i++) {
                if (i > s->coded_fragment_list_index)
                    return -1;

                if (s->all_fragments[s->coded_fragment_list[i]].qpi == qpi) {
                    s->all_fragments[s->coded_fragment_list[i]].qpi += bit;
                    j++;
                }
            }

            if (run_length == 4129)
                bit = get_bits1(gb);
