    *Neither the name of the University of Cambridge nor the names of
     its contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
The references are:
 * Machine learning for high-speed corner detection,
   E. Rosten and T. Drummond, ECCV 2006
 * Faster and better: A machine learning approach to corner detection
   E. Rosten, R. Porter and T. Drummond, PAMI, 2009
*/

#include "precomp.hpp"
#include "fast.hpp"
#include "fast_score.hpp"
#include "opencl_kernels_features2d.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

namespace cv
{

template<int patternSize>
void FAST_t(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    Mat img = _img.getMat();
    const int K = patternSize/2, N = patternSize + K + 1;
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step, patternSize);

#if CV_SIMD128
    const int quarterPatternSize = patternSize/4;
    v_uint8x16 delta = v_setall_u8(0x80), t = v_setall_u8((char)threshold), K16 = v_setall_u8((char)K);
#if CV_TRY_AVX2
    Ptr<opt_AVX2::FAST_t_patternSize16_AVX2> fast_t_impl_avx2;
    if(CV_CPU_HAS_SUPPORT_AVX2)
        fast_t_impl_avx2 = opt_AVX2::FAST_t_patternSize16_AVX2::getImpl(img.cols, threshold, nonmax_suppression, pixel);
#endif

#endif

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    AutoBuffer<uchar> _buf((img.cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128);
    uchar* buf[3];
    buf[0] = _buf.data(); buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
    int* cpbuf[3];
    cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    memset(buf[0], 0, img.cols*3);

    for(i = 3; i < img.rows-2; i++)
    {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
        {
            j = 3;
#if CV_SIMD128
            {
                if( patternSize == 16 )
                {
#if CV_TRY_AVX2
                    if (fast_t_impl_avx2)
                        fast_t_impl_avx2->process(j, ptr, curr, cornerpos, ncorners);
#endif
                    //vz if (j <= (img.cols - 27)) //it doesn't make sense using vectors for less than 8 elements
                    {
                        for (; j < img.cols - 16 - 3; j += 16, ptr += 16)
                        {
                            v_uint8x16 v = v_load(ptr);
                            v_int8x16 v0 = v_reinterpret_as_s8((v + t) ^ delta);
                            v_int8x16 v1 = v_reinterpret_as_s8((v - t) ^ delta);

                            v_int8x16 x0 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[0]), delta));
                            v_int8x16 x1 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[quarterPatternSize]), delta));
                            v_int8x16 x2 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[2*quarterPatternSize]), delta));
                            v_int8x16 x3 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[3*quarterPatternSize]), delta));

                            v_int8x16 m0, m1;
                            m0 = (v0 < x0) & (v0 < x1);
                            m1 = (x0 < v1) & (x1 < v1);
                            m0 = m0 | ((v0 < x1) & (v0 < x2));
                            m1 = m1 | ((x1 < v1) & (x2 < v1));
                            m0 = m0 | ((v0 < x2) & (v0 < x3));
                            m1 = m1 | ((x2 < v1) & (x3 < v1));
                            m0 = m0 | ((v0 < x3) & (v0 < x0));
                            m1 = m1 | ((x3 < v1) & (x0 < v1));
                            m0 = m0 | m1;

                            if( !v_check_any(m0) )
                                continue;
                            if( !v_check_any(v_combine_low(m0, m0)) )
                            {
                                j -= 8;
                                ptr -= 8;
                                continue;
                            }

                            v_int8x16 c0 = v_setzero_s8();
                            v_int8x16 c1 = v_setzero_s8();
                            v_uint8x16 max0 = v_setzero_u8();
                            v_uint8x16 max1 = v_setzero_u8();
                            for( k = 0; k < N; k++ )
                            {
                                v_int8x16 x = v_reinterpret_as_s8(v_load((ptr + pixel[k])) ^ delta);
                                m0 = v0 < x;
                                m1 = x < v1;

                                c0 = v_sub_wrap(c0, m0) & m0;
                                c1 = v_sub_wrap(c1, m1) & m1;

                                max0 = v_max(max0, v_reinterpret_as_u8(c0));
                                max1 = v_max(max1, v_reinterpret_as_u8(c1));
                            }

                            max0 = K16 < v_max(max0, max1);
                            int m = -v_reduce_sum(v_reinterpret_as_s8(max0));
                            uchar mflag[16];
                            v_store(mflag, max0);

                            for( k = 0; m > 0 && k < 16; k++ )
                            {
                                if(mflag[k])
                                {
                                    --m;
                                    cornerpos[ncorners++] = j+k;
                                    if(nonmax_suppression)
                                    {
                                        short d[25];
                                        for (int _k = 0; _k < 25; _k++)
                                            d[_k] = (short)(ptr[k] - ptr[k + pixel[_k]]);

                                        v_int16x8 a0, b0, a1, b1;
                                        a0 = b0 = a1 = b1 = v_load(d + 8);
                                        for(int shift = 0; shift < 8; ++shift)
                                        {
                                            v_int16x8 v_nms = v_load(d + shift);
                                            a0 = v_min(a0, v_nms);
                                            b0 = v_max(b0, v_nms);
                                            v_nms = v_load(d + 9 + shift);
                                            a1 = v_min(a1, v_nms);
                                            b1 = v_max(b1, v_nms);
                                        }
                                        curr[j + k] = (uchar)(v_reduce_max(v_max(v_max(a0, a1), v_setzero_s16() - v_min(b0, b1))) - 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
#endif
            for( ; j < img.cols - 3; j++, ptr++ )
            {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                {
                    int vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x < vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }

                if( d & 2 )
                {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
