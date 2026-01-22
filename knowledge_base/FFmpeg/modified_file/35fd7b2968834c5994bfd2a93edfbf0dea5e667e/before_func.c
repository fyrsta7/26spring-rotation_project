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
