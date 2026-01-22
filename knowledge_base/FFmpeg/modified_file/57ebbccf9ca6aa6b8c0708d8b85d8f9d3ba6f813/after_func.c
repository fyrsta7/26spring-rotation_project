
#undef NDEBUG
#include <assert.h>

#define OGGVORBIS_FRAME_SIZE 64

#define BUFFER_SIZE (1024*64)

typedef struct OggVorbisContext {
    vorbis_info vi ;
    vorbis_dsp_state vd ;
    vorbis_block vb ;
    uint8_t buffer[BUFFER_SIZE];
    int buffer_index;
    int eof;

    /* decoder */
    vorbis_comment vc ;
    ogg_packet op;
} OggVorbisContext ;


static av_cold int oggvorbis_init_encoder(vorbis_info *vi, AVCodecContext *avccontext) {
    double cfreq;

    if(avccontext->flags & CODEC_FLAG_QSCALE) {
        /* variable bitrate */
        if(vorbis_encode_setup_vbr(vi, avccontext->channels,
                avccontext->sample_rate,
                avccontext->global_quality / (float)FF_QP2LAMBDA / 10.0))
            return -1;
    } else {
        int minrate = avccontext->rc_min_rate > 0 ? avccontext->rc_min_rate : -1;
