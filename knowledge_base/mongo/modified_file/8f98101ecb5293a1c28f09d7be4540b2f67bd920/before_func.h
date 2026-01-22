*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU Affero General Public License for more details.
*
*    You should have received a copy of the GNU Affero General Public License
*    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "namespace.h"

namespace mongo { 

    inline Namespace& Namespace::operator=(const char *ns) {
        // we fill the remaining space with all zeroes here.  as the full Namespace struct is in 
        // the datafiles (the .ns files specifically), that is helpful as then they are deterministic 
        // in the bytes they have for a given sequence of operations.  that makes testing and debugging
        // the data files easier.
        //
