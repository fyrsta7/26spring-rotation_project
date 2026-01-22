#include "libavutil/intreadwrite.h"
#include "bswapdsp.h"
#include "canopus.h"
#include "get_bits.h"
#include "avcodec.h"
#include "internal.h"
#include "thread.h"

#define VLC_BITS 7
#define VLC_DEPTH 2


typedef struct CLLCContext {
    AVCodecContext *avctx;
    BswapDSPContext bdsp;

    uint8_t *swapped_buf;
    int      swapped_buf_size;
} CLLCContext;

static int read_code_table(CLLCContext *ctx, GetBitContext *gb, VLC *vlc)
{
    uint8_t symbols[256];
    uint8_t bits[256];
    int num_lens, num_codes, num_codes_sum;
    int i, j, count;

    count         = 0;
    num_codes_sum = 0;

    num_lens = get_bits(gb, 5);

    if (num_lens > VLC_BITS * VLC_DEPTH) {
        av_log(ctx->avctx, AV_LOG_ERROR, "To long VLCs %d\n", num_lens);
        return AVERROR_INVALIDDATA;
    }

    for (i = 0; i < num_lens; i++) {
