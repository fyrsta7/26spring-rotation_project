    void (*saturate_output)(struct cook *q, float *out);

    AVCodecContext*     avctx;
    AudioDSPContext     adsp;
    GetBitContext       gb;
    /* stream data */
    int                 num_vectors;
    int                 samples_per_channel;
