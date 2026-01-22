    void (*saturate_output)(struct cook *q, float *out);

    AVCodecContext*     avctx;
    AudioDSPContext     adsp;
    GetBitContext       gb;
    /* stream data */
    int                 num_vectors;
    int                 samples_per_channel;
    /* states */
    AVLFG               random_state;
    int                 discarded_packets;

    /* transform data */
    FFTContext          mdct_ctx;
    float*              mlt_window;
