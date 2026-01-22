 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <LibGfx/DisjointRectSet.h>

namespace Gfx {

bool DisjointRectSet::add_no_shatter(const IntRect& new_rect)
{
    if (new_rect.is_empty())
        return false;
    for (auto& rect : m_rects) {
        if (rect.contains(new_rect))
            return false;
    }

    m_rects.append(new_rect);
    return true;
}

void DisjointRectSet::shatter()
{
    Vector<IntRect, 32> output;
    output.ensure_capacity(m_rects.size());
    bool pass_had_intersections = false;
    do {
        pass_had_intersections = false;
