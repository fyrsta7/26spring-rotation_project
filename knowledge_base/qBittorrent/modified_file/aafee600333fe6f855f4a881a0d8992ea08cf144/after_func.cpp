 * and distribute the linked executables. You must obey the GNU General Public
 * License in all respects for all of the code used other than "OpenSSL".  If you
 * modify file(s), you may extend this exception to your version of the file(s),
 * but you are not obligated to do so. If you do not wish to do so, delete this
 * exception statement from your version.
 */

#include "ltqbitarray.h"

#include <memory>

#include <libtorrent/bitfield.hpp>

#include <QBitArray>

namespace
{
    unsigned char reverseByte(const unsigned char byte)
    {
        // https://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64Bits
        return (((byte * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL) >> 32;
    }
}

