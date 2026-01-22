 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <folly/detail/SingletonStackTrace.h>

#include <folly/experimental/symbolizer/ElfCache.h>
#include <folly/experimental/symbolizer/Symbolizer.h>
#include <folly/portability/Config.h>

namespace folly {
namespace detail {

std::string getSingletonStackTrace() {
#if FOLLY_HAVE_ELF && FOLLY_HAVE_DWARF

  // Get and symbolize stack trace
  constexpr size_t kMaxStackTraceDepth = 100;
  auto addresses =
      std::make_unique<symbolizer::FrameArray<kMaxStackTraceDepth>>();

  if (!getStackTraceSafe(*addresses)) {
    return "";
  } else {
    symbolizer::ElfCache elfCache;

