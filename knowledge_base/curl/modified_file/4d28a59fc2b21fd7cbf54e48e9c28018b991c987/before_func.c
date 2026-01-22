 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/
#include "tool_setup.h"

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_cfgable.h"
#include "tool_cb_prg.h"

#include "memdebug.h" /* keep this as LAST include */

/*
** callback for CURLOPT_PROGRESSFUNCTION
*/

#define MAX_BARLENGTH 256

int tool_progress_cb(void *clientp,
                     double dltotal, double dlnow,
                     double ultotal, double ulnow)
{
  /* The original progress-bar source code was written for curl by Lars Aas,
     and this new edition inherits some of his concepts. */

  char line[MAX_BARLENGTH+1];
  char format[40];
  double frac;
  double percent;
  int barwidth;
  int num;
  int i;

  struct ProgressData *bar = (struct ProgressData *)clientp;

  /* expected transfer size */
  curl_off_t total = (curl_off_t)dltotal + (curl_off_t)ultotal +
    bar->initial_size;

  /* we've come this far */
  curl_off_t point = (curl_off_t)dlnow + (curl_off_t)ulnow +
    bar->initial_size;

  if(point > total)
    /* we have got more than the expected total! */
    total = point;

  /* simply count invokes */
  bar->calls++;

  if(total < 1) {
    curl_off_t prevblock = bar->prev / 1024;
    curl_off_t thisblock = point / 1024;
