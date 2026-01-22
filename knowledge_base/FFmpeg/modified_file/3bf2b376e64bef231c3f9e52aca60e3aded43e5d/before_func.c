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

#include "libavutil/intreadwrite.h"

#include "avformat.h"
#include "rawdec.h"


static int dvbsub_probe(AVProbeData *p)
{
    int i, j, k;
    const uint8_t *end = p->buf + p->buf_size;
    int type, page_id, len;
    int max_score = 0;

    for(i=0; i<p->buf_size; i++){
        const uint8_t *ptr = p->buf + i;
        uint8_t histogram[6] = {0};
        int min = 255;
        for(j=0; ptr + 6 < end; j++) {
            if (*ptr != 0x0f)
                break;
            type    = ptr[1];
            page_id = AV_RB16(ptr + 2);
            len     = AV_RB16(ptr + 4);
            if (type == 0x80) {
