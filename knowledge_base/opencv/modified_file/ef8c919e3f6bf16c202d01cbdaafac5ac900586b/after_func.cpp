//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "undistort.hpp"

namespace cv
{

int initUndistortRectifyMapLine_AVX(float* m1f, float* m2f, short* m1, ushort* m2, double* matTilt, const double* ir,
                                    double& _x, double& _y, double& _w, int width, int m1type,
                                    double& k1, double& k2, double& k3, double& k4, double& k5, double& k6,
                                    double& p1, double& p2, double& s1, double& s2, double& s3, double& s4,
                                    double& u0, double& v0, double& fx, double& fy)
{
    int j = 0;

    static const __m256d __one = _mm256_set1_pd(1.0);
    static const __m256d __two = _mm256_set1_pd(2.0);

    const __m256d __matTilt_00 = _mm256_set1_pd(matTilt[0]);
    const __m256d __matTilt_10 = _mm256_set1_pd(matTilt[3]);
    const __m256d __matTilt_20 = _mm256_set1_pd(matTilt[6]);

    const __m256d __matTilt_01 = _mm256_set1_pd(matTilt[1]);
    const __m256d __matTilt_11 = _mm256_set1_pd(matTilt[4]);
    const __m256d __matTilt_21 = _mm256_set1_pd(matTilt[7]);

    const __m256d __matTilt_02 = _mm256_set1_pd(matTilt[2]);
    const __m256d __matTilt_12 = _mm256_set1_pd(matTilt[5]);
    const __m256d __matTilt_22 = _mm256_set1_pd(matTilt[8]);

    for (; j <= width - 4; j += 4, _x += 4 * ir[0], _y += 4 * ir[3], _w += 4 * ir[6])
    {
        // Question: Should we load the constants first?
        __m256d __w = _mm256_div_pd(__one, _mm256_set_pd(_w + 3 * ir[6], _w + 2 * ir[6], _w + ir[6], _w));
        __m256d __x = _mm256_mul_pd(_mm256_set_pd(_x + 3 * ir[0], _x + 2 * ir[0], _x + ir[0], _x), __w);
        __m256d __y = _mm256_mul_pd(_mm256_set_pd(_y + 3 * ir[3], _y + 2 * ir[3], _y + ir[3], _y), __w);
        __m256d __x2 = _mm256_mul_pd(__x, __x);
        __m256d __y2 = _mm256_mul_pd(__y, __y);
        __m256d __r2 = _mm256_add_pd(__x2, __y2);
        __m256d __2xy = _mm256_mul_pd(__two, _mm256_mul_pd(__x, __y));
        __m256d __kr = _mm256_div_pd(
#if CV_FMA3
            _mm256_fmadd_pd(_mm256_fmadd_pd(_mm256_fmadd_pd(_mm256_set1_pd(k3), __r2, _mm256_set1_pd(k2)), __r2, _mm256_set1_pd(k1)), __r2, __one),
            _mm256_fmadd_pd(_mm256_fmadd_pd(_mm256_fmadd_pd(_mm256_set1_pd(k6), __r2, _mm256_set1_pd(k5)), __r2, _mm256_set1_pd(k4)), __r2, __one)
#else
            _mm256_add_pd(__one, _mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(k3), __r2), _mm256_set1_pd(k2)), __r2), _mm256_set1_pd(k1)), __r2)),
            _mm256_add_pd(__one, _mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(k6), __r2), _mm256_set1_pd(k5)), __r2), _mm256_set1_pd(k4)), __r2))
#endif
        );
        __m256d __r22 = _mm256_mul_pd(__r2, __r2);
#if CV_FMA3
        __m256d __xd = _mm256_fmadd_pd(__x, __kr,
            _mm256_add_pd(
                _mm256_fmadd_pd(_mm256_set1_pd(p1), __2xy, _mm256_mul_pd(_mm256_set1_pd(p2), _mm256_fmadd_pd(__two, __x2, __r2))),
                _mm256_fmadd_pd(_mm256_set1_pd(s1), __r2, _mm256_mul_pd(_mm256_set1_pd(s2), __r22))));
        __m256d __yd = _mm256_fmadd_pd(__y, __kr,
            _mm256_add_pd(
                _mm256_fmadd_pd(_mm256_set1_pd(p1), _mm256_fmadd_pd(__two, __y2, __r2), _mm256_mul_pd(_mm256_set1_pd(p2), __2xy)),
                _mm256_fmadd_pd(_mm256_set1_pd(s3), __r2, _mm256_mul_pd(_mm256_set1_pd(s4), __r22))));

        __m256d __vecTilt2 = _mm256_fmadd_pd(__matTilt_20, __xd, _mm256_fmadd_pd(__matTilt_21, __yd, __matTilt_22));
#else
        __m256d __xd = _mm256_add_pd(
            _mm256_mul_pd(__x, __kr),
            _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(_mm256_set1_pd(p1), __2xy),
                    _mm256_mul_pd(_mm256_set1_pd(p2), _mm256_add_pd(__r2, _mm256_mul_pd(__two, __x2)))),
                _mm256_add_pd(
                    _mm256_mul_pd(_mm256_set1_pd(s1), __r2),
                    _mm256_mul_pd(_mm256_set1_pd(s2), __r22))));
        __m256d __yd = _mm256_add_pd(
            _mm256_mul_pd(__y, __kr),
            _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(_mm256_set1_pd(p1), _mm256_add_pd(__r2, _mm256_mul_pd(__two, __y2))),
                    _mm256_mul_pd(_mm256_set1_pd(p2), __2xy)),
                _mm256_add_pd(
                    _mm256_mul_pd(_mm256_set1_pd(s3), __r2),
                    _mm256_mul_pd(_mm256_set1_pd(s4), __r22))));

        __m256d __vecTilt2 = _mm256_add_pd(_mm256_add_pd(
            _mm256_mul_pd(__matTilt_20, __xd), _mm256_mul_pd(__matTilt_21, __yd)), __matTilt_22);
#endif
        __m256d __invProj = _mm256_blendv_pd(
            __one, _mm256_div_pd(__one, __vecTilt2),
            _mm256_cmp_pd(__vecTilt2, _mm256_setzero_pd(), _CMP_EQ_OQ));

#if CV_FMA3
        __m256d __u = _mm256_fmadd_pd(__matTilt_00, __xd, _mm256_fmadd_pd(__matTilt_01, __yd, __matTilt_02));
        __u = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_set1_pd(fx), __invProj), __u, _mm256_set1_pd(u0));

        __m256d __v = _mm256_fmadd_pd(__matTilt_10, __xd, _mm256_fmadd_pd(__matTilt_11, __yd, __matTilt_12));
        __v = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_set1_pd(fy), __invProj), __v, _mm256_set1_pd(v0));
#else
        __m256d __u = _mm256_add_pd(_mm256_add_pd(
            _mm256_mul_pd(__matTilt_00, __xd), _mm256_mul_pd(__matTilt_01, __yd)), __matTilt_02);
        __u = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(fx), __invProj), __u), _mm256_set1_pd(u0));

        __m256d __v = _mm256_add_pd(_mm256_add_pd(
            _mm256_mul_pd(__matTilt_10, __xd), _mm256_mul_pd(__matTilt_11, __yd)), __matTilt_12);
        __v = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(fy), __invProj), __v), _mm256_set1_pd(v0));
#endif

        if (m1type == CV_32FC1)
        {
            _mm_storeu_ps(&m1f[j], _mm256_cvtpd_ps(__u));
            _mm_storeu_ps(&m2f[j], _mm256_cvtpd_ps(__v));
        }
