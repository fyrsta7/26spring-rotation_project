 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Libav; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "config.h"
#include "libavutil/attributes.h"
#include "libavutil/intreadwrite.h"
#include "dcadsp.h"

static void int8x8_fmul_int32_c(float *dst, const int8_t *src, int scale)
{
    float fscale = scale / 16.0;
    int i;
    for (i = 0; i < 8; i++)
        dst[i] = src[i] * fscale;
}

static inline void
