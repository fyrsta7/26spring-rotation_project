
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libavutil/imgutils.h"

#include "avcodec.h"
#include "bytestream.h"
#include "internal.h"

#include <zlib.h>

typedef struct WCMVContext {
    int         bpp;
    z_stream    zstream;
    AVFrame    *prev_frame;
    uint8_t     block_data[65536*8];
} WCMVContext;

static int decode_frame(AVCodecContext *avctx,
                        void *data, int *got_frame,
                        AVPacket *avpkt)
{
    WCMVContext *s = avctx->priv_data;
    AVFrame *frame = data;
    int skip, blocks, zret, ret, intra = 0, bpp = s->bpp;
    GetByteContext gb;
    uint8_t *dst;

    ret = inflateReset(&s->zstream);
    if (ret != Z_OK) {
        av_log(avctx, AV_LOG_ERROR, "Inflate reset error: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    bytestream2_init(&gb, avpkt->data, avpkt->size);

    if ((ret = ff_get_buffer(avctx, frame, AV_GET_BUFFER_FLAG_REF)) < 0)
        return ret;

    blocks = bytestream2_get_le16(&gb);
    if (blocks > 5) {
        GetByteContext bgb;
        int x = 0, size;

        if (blocks * 8 >= 0xFFFF)
            size = bytestream2_get_le24(&gb);
        else if (blocks * 8 >= 0xFF)
            size = bytestream2_get_le16(&gb);
        else
            size = bytestream2_get_byte(&gb);

        skip = bytestream2_tell(&gb);
        if (size > avpkt->size - skip)
            return AVERROR_INVALIDDATA;

        s->zstream.next_in  = avpkt->data + skip;
        s->zstream.avail_in = size;
        s->zstream.next_out  = s->block_data;
        s->zstream.avail_out = sizeof(s->block_data);

        zret = inflate(&s->zstream, Z_FINISH);
        if (zret != Z_STREAM_END) {
            av_log(avctx, AV_LOG_ERROR,
                   "Inflate failed with return code: %d.\n", zret);
            return AVERROR_INVALIDDATA;
        }

        ret = inflateReset(&s->zstream);
        if (ret != Z_OK) {
            av_log(avctx, AV_LOG_ERROR, "Inflate reset error: %d\n", ret);
            return AVERROR_EXTERNAL;
        }

        bytestream2_skip(&gb, size);
        bytestream2_init(&bgb, s->block_data, blocks * 8);

        for (int i = 0; i < blocks; i++) {
            int w, h;

            bytestream2_skip(&bgb, 4);
            w = bytestream2_get_le16(&bgb);
            h = bytestream2_get_le16(&bgb);
            if (x + bpp * (int64_t)w * h > INT_MAX)
                return AVERROR_INVALIDDATA;
            x += bpp * w * h;
        }

        if (x >= 0xFFFF)
            bytestream2_skip(&gb, 3);
        else if (x >= 0xFF)
            bytestream2_skip(&gb, 2);
        else
            bytestream2_skip(&gb, 1);

        skip = bytestream2_tell(&gb);

        s->zstream.next_in  = avpkt->data + skip;
        s->zstream.avail_in = avpkt->size - skip;

        bytestream2_init(&gb, s->block_data, blocks * 8);
    } else if (blocks) {
        int x = 0;

        bytestream2_seek(&gb, 2, SEEK_SET);

        for (int i = 0; i < blocks; i++) {
            int w, h;

            bytestream2_skip(&gb, 4);
            w = bytestream2_get_le16(&gb);
            h = bytestream2_get_le16(&gb);
            if (x + bpp * (int64_t)w * h > INT_MAX)
                return AVERROR_INVALIDDATA;
            x += bpp * w * h;
        }

        if (x >= 0xFFFF)
            bytestream2_skip(&gb, 3);
        else if (x >= 0xFF)
            bytestream2_skip(&gb, 2);
        else
            bytestream2_skip(&gb, 1);

        skip = bytestream2_tell(&gb);

        s->zstream.next_in  = avpkt->data + skip;
        s->zstream.avail_in = avpkt->size - skip;

        bytestream2_seek(&gb, 2, SEEK_SET);
    }

    if (s->prev_frame->data[0]) {
        ret = av_frame_copy(frame, s->prev_frame);
        if (ret < 0)
            return ret;
    } else {
        ptrdiff_t linesize[4] = { frame->linesize[0], 0, 0, 0 };
        av_image_fill_black(frame->data, linesize, avctx->pix_fmt, 0,
                            avctx->width, avctx->height);
    }

    for (int block = 0; block < blocks; block++) {
        int x, y, w, h;

        x = bytestream2_get_le16(&gb);
        y = bytestream2_get_le16(&gb);
        w = bytestream2_get_le16(&gb);
        h = bytestream2_get_le16(&gb);

        if (blocks == 1 && x == 0 && y == 0 && w == avctx->width && h == avctx->height)
            intra = 1;

        if (x + w > avctx->width || y + h > avctx->height)
            return AVERROR_INVALIDDATA;

        if (w > avctx->width || h > avctx->height)
            return AVERROR_INVALIDDATA;

        dst = frame->data[0] + (avctx->height - y - 1) * frame->linesize[0] + x * bpp;
        for (int i = 0; i < h; i++) {
            s->zstream.next_out  = dst;
            s->zstream.avail_out = w * bpp;

            zret = inflate(&s->zstream, Z_SYNC_FLUSH);
            if (zret != Z_OK && zret != Z_STREAM_END) {
