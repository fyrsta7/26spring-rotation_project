 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * VBLE Decoder
 */

#define ALT_BITSTREAM_READER_LE

#include "avcodec.h"
#include "get_bits.h"

typedef struct {
    AVCodecContext *avctx;

    int            size;
    int            flags;
    uint8_t        *len;
    uint8_t        *val;
} VBLEContext;

static uint8_t vble_read_reverse_unary(GetBitContext *gb)
{
    static const uint8_t LUT[256] = {
        8,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
        5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
        6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
