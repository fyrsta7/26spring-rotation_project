 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "redis.h"

/* Dictionary type for latency events. Key/Val destructors are set to NULL
 * since we never delete latency time series at runtime. */
int dictStringKeyCompare(void *privdata, const void *key1, const void *key2) {
    return strcmp(key1,key2) == 0;
}

unsigned int dictStringHash(const void *key) {
    return dictGenHashFunction(key, strlen(key));
}

dictType latencyTimeSeriesDictType = {
    dictStringHash,             /* hash function */
