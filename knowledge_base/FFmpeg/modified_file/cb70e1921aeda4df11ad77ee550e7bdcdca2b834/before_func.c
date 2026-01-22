 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "avcodec.h"
#include "internal.h"
#include "mathops.h"

static int xbm_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                            const AVFrame *p, int *got_packet)
{
    int i, j, commas, ret, size, linesize;
    uint8_t *ptr, *buf;

    linesize = (avctx->width + 7) / 8;
    commas   = avctx->height * linesize;
    size     = avctx->height * (linesize * 7 + 2) + 109;
    if ((ret = ff_alloc_packet2(avctx, pkt, size, 0)) < 0)
        return ret;

    buf = pkt->data;
    ptr = p->data[0];
