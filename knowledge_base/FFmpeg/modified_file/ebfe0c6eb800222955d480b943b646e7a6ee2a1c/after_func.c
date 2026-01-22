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

#include "libavutil/crc.h"
#include "libavcodec/ac3_parser.h"
#include "avformat.h"
#include "rawdec.h"

static int ac3_eac3_probe(AVProbeData *p, enum CodecID expected_codec_id)
{
    int max_frames, first_frames = 0, frames;
    uint8_t *buf, *buf2, *end;
    AC3HeaderInfo hdr;
    GetBitContext gbc;
    enum CodecID codec_id = CODEC_ID_AC3;

    max_frames = 0;
    buf = p->buf;
    end = buf + p->buf_size;

    for(; buf < end; buf++) {
        if(buf > p->buf && (buf[0] != 0x0B || buf[1] != 0x77) )
            continue;
        buf2 = buf;

        for(frames = 0; buf2 < end; frames++) {
            if(!memcmp(buf2, "\x1\x10\0\0\0\0\0\0", 8))
                buf2+=16;
            init_get_bits(&gbc, buf2, 54);
            if(avpriv_ac3_parse_header(&gbc, &hdr) < 0)
                break;
            if(buf2 + hdr.frame_size > end ||
