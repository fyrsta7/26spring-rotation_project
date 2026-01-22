                }
                n = s->block_len - s->coefs_end[bsize];
                for(i = 0;i < n; i++)
                    *coefs++ = 0.0;
            }
        }
    }

#ifdef TRACE
    for(ch = 0; ch < s->nb_channels; ch++) {
        if (s->channel_coded[ch]) {
            dump_floats(s, "exponents", 3, s->exponents[ch], s->block_len);
            dump_floats(s, "coefs", 1, s->coefs[ch], s->block_len);
        }
    }
#endif

    if (s->ms_stereo && s->channel_coded[1]) {
        /* nominal case for ms stereo: we do it before mdct */
        /* no need to optimize this case because it should almost
           never happen */
        if (!s->channel_coded[0]) {
            tprintf(s->avctx, "rare ms-stereo case happened\n");
            memset(s->coefs[0], 0, sizeof(float) * s->block_len);
            s->channel_coded[0] = 1;
        }

        s->dsp.butterflies_float(s->coefs[0], s->coefs[1], s->block_len);
    }

next:
    for(ch = 0; ch < s->nb_channels; ch++) {
        int n4, index;

        n4 = s->block_len / 2;
        if(s->channel_coded[ch]){
            ff_imdct_calc(&s->mdct_ctx[bsize], s->output, s->coefs[ch]);
        }else if(!(s->ms_stereo && ch==1))
            memset(s->output, 0, sizeof(s->output));

        /* multiply by the window and add in the frame */
        index = (s->frame_len / 2) + s->block_pos - n4;
        wma_window(s, &s->frame_out[ch][index]);
    }

    /* update block number */
    s->block_num++;
    s->block_pos += s->block_len;
    if (s->block_pos >= s->frame_len)
        return 1;
    else
        return 0;
}
