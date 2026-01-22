 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file fft.c
 * FFT/IFFT transforms.
 */

#include "dsputil.h"

/**
 * The size of the FFT is 2^nbits. If inverse is TRUE, inverse FFT is
 * done
 */
int ff_fft_init(FFTContext *s, int nbits, int inverse)
{
    int i, j, m, n;
    float alpha, c1, s1, s2;

    s->nbits = nbits;
    n = 1 << nbits;

    s->exptab = av_malloc((n / 2) * sizeof(FFTComplex));
    if (!s->exptab)
        goto fail;
    s->revtab = av_malloc(n * sizeof(uint16_t));
    if (!s->revtab)
        goto fail;
    s->inverse = inverse;

    s2 = inverse ? 1.0 : -1.0;

    for(i=0;i<(n/2);i++) {
        alpha = 2 * M_PI * (float)i / (float)n;
        c1 = cos(alpha);
        s1 = sin(alpha) * s2;
        s->exptab[i].re = c1;
        s->exptab[i].im = s1;
    }
    s->fft_calc = ff_fft_calc_c;
    s->imdct_calc = ff_imdct_calc;
    s->exptab1 = NULL;

    /* compute constant table for HAVE_SSE version */
#if defined(HAVE_MMX) \
    || (defined(HAVE_ALTIVEC) && !defined(ALTIVEC_USE_REFERENCE_C_CODE))
    {
        int has_vectors = mm_support();

        if (has_vectors) {
#if defined(HAVE_MMX)
            if (has_vectors & MM_3DNOWEXT)
                s->imdct_calc = ff_imdct_calc_3dn2;
            if (has_vectors & MM_3DNOWEXT)
                /* 3DNowEx for Athlon(XP) */
                s->fft_calc = ff_fft_calc_3dn2;
            else if (has_vectors & MM_3DNOW)
                /* 3DNow! for K6-2/3 */
                s->fft_calc = ff_fft_calc_3dn;
            if (has_vectors & MM_SSE2)
                /* SSE for P4/K8 */
                s->fft_calc = ff_fft_calc_sse;
            else if ((has_vectors & MM_SSE) &&
                     s->fft_calc == ff_fft_calc_c)
                /* SSE for P3 */
                s->fft_calc = ff_fft_calc_sse;
#else /* HAVE_MMX */
            if (has_vectors & MM_ALTIVEC)
                s->fft_calc = ff_fft_calc_altivec;
#endif
        }
        if (s->fft_calc != ff_fft_calc_c) {
            int np, nblocks, np2, l;
            FFTComplex *q;

            np = 1 << nbits;
            nblocks = np >> 3;
            np2 = np >> 1;
            s->exptab1 = av_malloc(np * 2 * sizeof(FFTComplex));
            if (!s->exptab1)
                goto fail;
            q = s->exptab1;
            do {
                for(l = 0; l < np2; l += 2 * nblocks) {
                    *q++ = s->exptab[l];
                    *q++ = s->exptab[l + nblocks];

                    q->re = -s->exptab[l].im;
                    q->im = s->exptab[l].re;
                    q++;
                    q->re = -s->exptab[l + nblocks].im;
                    q->im = s->exptab[l + nblocks].re;
                    q++;
                }
