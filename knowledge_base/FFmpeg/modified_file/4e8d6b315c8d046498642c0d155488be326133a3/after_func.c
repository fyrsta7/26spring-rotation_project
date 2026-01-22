 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/**
 * @file
 * A very simple tv station logo remover
 * Originally imported from MPlayer libmpcodecs/vf_delogo.c,
 * the algorithm was later improved.
 */

#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

/**
 * Apply a simple delogo algorithm to the image in src and put the
 * result in dst.
 *
 * The algorithm is only applied to the region specified by the logo
 * parameters.
 *
 * @param w      width of the input image
 * @param h      height of the input image
 * @param logo_x x coordinate of the top left corner of the logo region
 * @param logo_y y coordinate of the top left corner of the logo region
 * @param logo_w width of the logo
 * @param logo_h height of the logo
 * @param band   the size of the band around the processed area
 * @param show   show a rectangle around the processed area, useful for
 *               parameters tweaking
 * @param direct if non-zero perform in-place processing
 */
static void apply_delogo(uint8_t *dst, int dst_linesize,
                         uint8_t *src, int src_linesize,
                         int w, int h, AVRational sar,
                         int logo_x, int logo_y, int logo_w, int logo_h,
                         int band, int show, int direct)
{
    int x, y, dist;
    uint64_t interp, weightl, weightr, weightt, weightb;
    uint8_t *xdst, *xsrc;

    uint8_t *topleft, *botleft, *topright;
    unsigned int left_sample, right_sample;
    int xclipl, xclipr, yclipt, yclipb;
    int logo_x1, logo_x2, logo_y1, logo_y2;

    xclipl = FFMAX(-logo_x, 0);
    xclipr = FFMAX(logo_x+logo_w-w, 0);
    yclipt = FFMAX(-logo_y, 0);
    yclipb = FFMAX(logo_y+logo_h-h, 0);

    logo_x1 = logo_x + xclipl;
    logo_x2 = logo_x + logo_w - xclipr;
    logo_y1 = logo_y + yclipt;
    logo_y2 = logo_y + logo_h - yclipb;

    topleft  = src+logo_y1     * src_linesize+logo_x1;
    topright = src+logo_y1     * src_linesize+logo_x2-1;
    botleft  = src+(logo_y2-1) * src_linesize+logo_x1;

    if (!direct)
        av_image_copy_plane(dst, dst_linesize, src, src_linesize, w, h);

    dst += (logo_y1 + 1) * dst_linesize;
    src += (logo_y1 + 1) * src_linesize;

    for (y = logo_y1+1; y < logo_y2-1; y++) {
        left_sample = topleft[src_linesize*(y-logo_y1)]   +
                      topleft[src_linesize*(y-logo_y1-1)] +
                      topleft[src_linesize*(y-logo_y1+1)];
        right_sample = topright[src_linesize*(y-logo_y1)]   +
                       topright[src_linesize*(y-logo_y1-1)] +
                       topright[src_linesize*(y-logo_y1+1)];

        for (x = logo_x1+1,
             xdst = dst+logo_x1+1,
             xsrc = src+logo_x1+1; x < logo_x2-1; x++, xdst++, xsrc++) {

            /* Weighted interpolation based on relative distances, taking SAR into account */
            weightl = (uint64_t)              (logo_x2-1-x) * (y-logo_y1) * (logo_y2-1-y) * sar.den;
            weightr = (uint64_t)(x-logo_x1)                 * (y-logo_y1) * (logo_y2-1-y) * sar.den;
