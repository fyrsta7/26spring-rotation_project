 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#pragma once

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

static inline uint64_t util_mul_div64(uint64_t num, uint64_t mul, uint64_t div)
{
#if !defined(_MSC_VER)
#if defined(__x86_64__)
	uint64_t rax, rdx;
	__asm__("mulq %2" : "=a"(rax), "=d"(rdx) : "r"(num), "a"(mul));
	__asm__("divq %1" : "=a"(rax) : "r"(div), "a"(rax), "d"(rdx));
	return rax;
