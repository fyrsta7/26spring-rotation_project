        sort_stt(s, s->state_transition);
    }

    if(s->version>1){
        s->num_h_slices=2;
        s->num_v_slices=2;
        write_extra_header(s);
    }

    if(init_slice_contexts(s) < 0)
        return -1;
    if(init_slice_state(s) < 0)
        return -1;

#define STATS_OUT_SIZE 1024*30
    if(avctx->flags & CODEC_FLAG_PASS1)
    avctx->stats_out= av_mallocz(STATS_OUT_SIZE);

    return 0;
}
#endif /* CONFIG_FFV1_ENCODER */


static void clear_state(FFV1Context *f){
