 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavcodec/imgconvert.h"
#include "avfilter.h"

/* TODO: buffer pool.  see comment for avfilter_default_get_video_buffer() */
static void avfilter_default_free_video_buffer(AVFilterPic *pic)
{
    av_free(pic->data[0]);
    av_free(pic);
}

/* TODO: set the buffer's priv member to a context structure for the whole
 * filter chain.  This will allow for a buffer pool instead of the constant
 * alloc & free cycle currently implemented. */
AVFilterPicRef *avfilter_default_get_video_buffer(AVFilterLink *link, int perms, int w, int h)
{
    AVFilterPic *pic = av_mallocz(sizeof(AVFilterPic));
    AVFilterPicRef *ref = av_mallocz(sizeof(AVFilterPicRef));
    int i, tempsize;
    char *buf;

    ref->pic   = pic;
    ref->w     = pic->w = w;
    ref->h     = pic->h = h;

