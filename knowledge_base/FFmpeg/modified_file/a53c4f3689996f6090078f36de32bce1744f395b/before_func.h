    int slice_rct_by_coef;
    int slice_rct_ry_coef;
} FFV1Context;

int ff_ffv1_common_init(AVCodecContext *avctx);
int ff_ffv1_init_slice_state(FFV1Context *f, FFV1Context *fs);
int ff_ffv1_init_slices_state(FFV1Context *f);
int ff_ffv1_init_slice_contexts(FFV1Context *f);
int ff_ffv1_allocate_initial_states(FFV1Context *f);
void ff_ffv1_clear_slice_state(FFV1Context *f, FFV1Context *fs);
int ff_ffv1_close(AVCodecContext *avctx);

static av_always_inline int fold(int diff, int bits)
{
    if (bits == 8)
        diff = (int8_t)diff;
    else {
        diff = sign_extend(diff, bits);
    }

    return diff;
}

static inline void update_vlc_state(VlcState *const state, const int v)
{
    int drift = state->drift;
    int count = state->count;
    state->error_sum += FFABS(v);
    drift            += v;

    if (count == 128) { // FIXME: variable
        count            >>= 1;
        drift            >>= 1;
