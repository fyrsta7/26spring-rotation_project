 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * QPEG codec.
 */

#include "avcodec.h"
#include "bytestream.h"
#include "internal.h"

typedef struct QpegContext{
    AVCodecContext *avctx;
    AVFrame *ref;
    uint32_t pal[256];
    GetByteContext buffer;
} QpegContext;

static void qpeg_decode_intra(QpegContext *qctx, uint8_t *dst,
                              int stride, int width, int height)
{
    int i;
    int code;
    int c0, c1;
    int run, copy;
    int filled = 0;
    int rows_to_go;

    rows_to_go = height;
    height--;
    dst = dst + height * stride;

    while ((bytestream2_get_bytes_left(&qctx->buffer) > 0) && (rows_to_go > 0)) {
        code = bytestream2_get_byte(&qctx->buffer);
        run = copy = 0;
        if(code == 0xFC) /* end-of-picture code */
            break;
        if(code >= 0xF8) { /* very long run */
            c0 = bytestream2_get_byte(&qctx->buffer);
            c1 = bytestream2_get_byte(&qctx->buffer);
            run = ((code & 0x7) << 16) + (c0 << 8) + c1 + 2;
        } else if (code >= 0xF0) { /* long run */
            c0 = bytestream2_get_byte(&qctx->buffer);
            run = ((code & 0xF) << 8) + c0 + 2;
        } else if (code >= 0xE0) { /* short run */
            run = (code & 0x1F) + 2;
        } else if (code >= 0xC0) { /* very long copy */
            c0 = bytestream2_get_byte(&qctx->buffer);
            c1 = bytestream2_get_byte(&qctx->buffer);
            copy = ((code & 0x3F) << 16) + (c0 << 8) + c1 + 1;
        } else if (code >= 0x80) { /* long copy */
            c0 = bytestream2_get_byte(&qctx->buffer);
            copy = ((code & 0x7F) << 8) + c0 + 1;
        } else { /* short copy */
            copy = code + 1;
        }

        /* perform actual run or copy */
        if(run) {
            int p;

            p = bytestream2_get_byte(&qctx->buffer);
            for(i = 0; i < run; i++) {
                int step = FFMIN(run - i, width - filled);
                memset(dst+filled, p, step);
                filled += step;
                i      += step - 1;
                if (filled >= width) {
                    filled = 0;
                    dst -= stride;
                    rows_to_go--;
                    while (run - i > width && rows_to_go > 0) {
                        memset(dst, p, width);
                        dst -= stride;
                        rows_to_go--;
                        i += width;
                    }
                    if(rows_to_go <= 0)
