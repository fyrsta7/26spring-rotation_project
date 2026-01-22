 * or later ("AGPL"), as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "tsdb.h"

// =============== PAGE-WISE FILE ===============
static int32_t tsdbOpenFile(const char *path, int32_t szPage, int32_t flag, STsdbFD **ppFD) {
  int32_t  code = 0;
  STsdbFD *pFD = NULL;

  *ppFD = NULL;

  pFD = (STsdbFD *)taosMemoryCalloc(1, sizeof(*pFD) + strlen(path) + 1);
  if (pFD == NULL) {
    code = TSDB_CODE_OUT_OF_MEMORY;
    goto _exit;
  }

  pFD->path = (char *)&pFD[1];
  strcpy(pFD->path, path);
  pFD->szPage = szPage;
  pFD->flag = flag;
  pFD->pFD = taosOpenFile(path, flag);
  if (pFD->pFD == NULL) {
    code = TAOS_SYSTEM_ERROR(errno);
    taosMemoryFree(pFD);
    goto _exit;
  }
  pFD->szPage = szPage;
  pFD->pgno = 0;
  pFD->pBuf = taosMemoryCalloc(1, szPage);
  if (pFD->pBuf == NULL) {
    code = TSDB_CODE_OUT_OF_MEMORY;
    taosCloseFile(&pFD->pFD);
    taosMemoryFree(pFD);
    goto _exit;
  }
  if (taosStatFile(path, &pFD->szFile, NULL) < 0) {
