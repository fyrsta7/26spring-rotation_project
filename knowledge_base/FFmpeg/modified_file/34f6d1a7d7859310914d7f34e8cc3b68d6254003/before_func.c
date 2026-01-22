 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"

#include "parser.h" //for ParseContext
#include "pnm.h"

typedef struct PNMParseContext {
    ParseContext pc;
    int remaining_bytes;
    int ascii_scan;
}PNMParseContext;

static int pnm_parse(AVCodecParserContext *s, AVCodecContext *avctx,
                     const uint8_t **poutbuf, int *poutbuf_size,
                     const uint8_t *buf, int buf_size)
{
    PNMParseContext *pnmpc = s->priv_data;
    ParseContext *pc = &pnmpc->pc;
    PNMContext pnmctx;
    int next = END_NOT_FOUND;
    int skip = 0;

    for (; pc->overread > 0; pc->overread--) {
        pc->buffer[pc->index++]= pc->buffer[pc->overread_index++];
    }

    if (pnmpc->remaining_bytes) {
        int inc = FFMIN(pnmpc->remaining_bytes, buf_size);
        skip += inc;
        pnmpc->remaining_bytes -= inc;

        if (!pnmpc->remaining_bytes)
            next = skip;
        goto end;
    }

retry:
    if (pc->index) {
        pnmctx.bytestream_start =
        pnmctx.bytestream       = pc->buffer;
        pnmctx.bytestream_end   = pc->buffer + pc->index;
    } else {
        pnmctx.bytestream_start =
        pnmctx.bytestream       = (uint8_t *) buf + skip; /* casts avoid warnings */
        pnmctx.bytestream_end   = (uint8_t *) buf + buf_size - skip;
    }
    if (ff_pnm_decode_header(avctx, &pnmctx) < 0) {
        if (pnmctx.bytestream < pnmctx.bytestream_end) {
            if (pc->index) {
                pc->index = 0;
                pnmpc->ascii_scan = 0;
            } else {
                unsigned step = FFMAX(1, pnmctx.bytestream - pnmctx.bytestream_start);

                skip += step;
            }
            goto retry;
        }
    } else if (pnmctx.type < 4) {
              uint8_t *bs  = pnmctx.bytestream;
        const uint8_t *end = pnmctx.bytestream_end;
        uint8_t *sync      = bs;

        if (pc->index) {
            av_assert0(pnmpc->ascii_scan <= end - bs);
            bs += pnmpc->ascii_scan;
        }

        while (bs < end) {
            int c;
            sync = bs;
            c = *bs++;
            if (c == '#')  {
                while (c != '\n' && bs < end)
                    c = *bs++;
            } else if (c == 'P') {
                next = bs - pnmctx.bytestream_start + skip - 1;
                pnmpc->ascii_scan = 0;
                break;
            }
        }
        if (next == END_NOT_FOUND)
            pnmpc->ascii_scan = sync - pnmctx.bytestream + skip;
    } else {
