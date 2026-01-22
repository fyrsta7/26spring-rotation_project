 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <stdlib.h>
#include <string.h>

#include "libavutil/imgutils.h"
#include "avcodec.h"
#include "internal.h"
#include "pnm.h"

static inline int pnm_space(int c)
{
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static void pnm_get(PNMContext *sc, char *str, int buf_size)
{
    char *s;
    int c;
    uint8_t *bs  = sc->bytestream;
    const uint8_t *end = sc->bytestream_end;

    /* skip spaces and comments */
    while (bs < end) {
