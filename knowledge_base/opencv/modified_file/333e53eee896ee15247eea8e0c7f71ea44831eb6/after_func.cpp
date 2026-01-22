       << "signedGradient" << signedGradient;
    if( !svmDetector.empty() )
        fs << "SVMDetector" << svmDetector;
    fs << "}";
}

bool HOGDescriptor::load(const String& filename, const String& objname)
{
    FileStorage fs(filename, FileStorage::READ);
    FileNode obj = !objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
    return read(obj);
}

void HOGDescriptor::save(const String& filename, const String& objName) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    write(fs, !objName.empty() ? objName : FileStorage::getDefaultObjectName(filename));
}

void HOGDescriptor::copyTo(HOGDescriptor& c) const
{
    c.winSize = winSize;
    c.blockSize = blockSize;
    c.blockStride = blockStride;
    c.cellSize = cellSize;
    c.nbins = nbins;
    c.derivAperture = derivAperture;
    c.winSigma = winSigma;
    c.histogramNormType = histogramNormType;
    c.L2HysThreshold = L2HysThreshold;
    c.gammaCorrection = gammaCorrection;
    c.svmDetector = svmDetector;
    c.nlevels = nlevels;
    c.signedGradient = signedGradient;
}

#if CV_NEON
// replace of _mm_set_ps
inline float32x4_t vsetq_f32(float f0, float f1, float f2, float f3)
{
    float32x4_t a = vdupq_n_f32(f0);
    a = vsetq_lane_f32(f1, a, 1);
    a = vsetq_lane_f32(f2, a, 2);
    a = vsetq_lane_f32(f3, a, 3);
    return a;
}
#endif
void HOGDescriptor::computeGradient(const Mat& img, Mat& grad, Mat& qangle,
    Size paddingTL, Size paddingBR) const
{
    CV_INSTRUMENT_REGION()

    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );

    Size gradsize(img.cols + paddingTL.width + paddingBR.width,
        img.rows + paddingTL.height + paddingBR.height);
    grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation

    Size wholeSize;
    Point roiofs;
    img.locateROI(wholeSize, roiofs);

    int i, x, y;
    int cn = img.channels();

    Mat_<float> _lut(1, 256);
    const float* const lut = &_lut(0,0);
#if CV_SSE2
    const int indeces[] = { 0, 1, 2, 3 };
    __m128i idx = _mm_loadu_si128((const __m128i*)indeces);
    __m128i ifour = _mm_set1_epi32(4);

    float* const _data = &_lut(0, 0);
    if( gammaCorrection )
        for( i = 0; i < 256; i += 4 )
        {
            _mm_storeu_ps(_data + i, _mm_sqrt_ps(_mm_cvtepi32_ps(idx)));
            idx = _mm_add_epi32(idx, ifour);
        }
    else
        for( i = 0; i < 256; i += 4 )
        {
            _mm_storeu_ps(_data + i, _mm_cvtepi32_ps(idx));
            idx = _mm_add_epi32(idx, ifour);
        }
#elif CV_NEON
    const int indeces[] = { 0, 1, 2, 3 };
    uint32x4_t idx = *(uint32x4_t*)indeces;
    uint32x4_t ifour = vdupq_n_u32(4);

    float* const _data = &_lut(0, 0);
    if( gammaCorrection )
        for( i = 0; i < 256; i++ )
            _lut(0,i) = std::sqrt((float)i);
    else
        for( i = 0; i < 256; i += 4 )
        {
            vst1q_f32(_data + i, vcvtq_f32_u32(idx));
            idx = vaddq_u32 (idx, ifour);
        }
#else
    if( gammaCorrection )
        for( i = 0; i < 256; i++ )
            _lut(0,i) = std::sqrt((float)i);
    else
        for( i = 0; i < 256; i++ )
            _lut(0,i) = (float)i;
#endif

    AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradsize.width + 2;

    const int borderType = (int)BORDER_REFLECT_101;

    for( x = -1; x < gradsize.width + 1; x++ )
        xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
        wholeSize.width, borderType) - roiofs.x;
    for( y = -1; y < gradsize.height + 1; y++ )
        ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
        wholeSize.height, borderType) - roiofs.y;

    // x- & y- derivatives for the whole row
    int width = gradsize.width;
    AutoBuffer<float> _dbuf(width*4);
    float* const dbuf = _dbuf;
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    if (cn == 3)
    {
        int end = gradsize.width + 2;
        xmap -= 1, x = 0;
#if CV_SSE2
        for ( ; x <= end - 4; x += 4)
        {
            __m128i mul_res = _mm_loadu_si128((const __m128i*)(xmap + x));
            mul_res = _mm_add_epi32(_mm_add_epi32(mul_res, mul_res), mul_res); // multiply by 3
            _mm_storeu_si128((__m128i*)(xmap + x), mul_res);
        }
#elif CV_NEON
        int32x4_t ithree = vdupq_n_s32(3);
        for ( ; x <= end - 4; x += 4)
            vst1q_s32(xmap + x, vmulq_s32(ithree, vld1q_s32(xmap + x)));
#endif
        for ( ; x < end; ++x)
            xmap[x] *= 3;
        xmap += 1;
    }

    float angleScale = signedGradient ? (float)(nbins/(2.0*CV_PI)) : (float)(nbins/CV_PI);
    for( y = 0; y < gradsize.height; y++ )
    {
        const uchar* imgPtr  = img.ptr(ymap[y]);
        //In case subimage is used ptr() generates an assert for next and prev rows
        //(see http://code.opencv.org/issues/4149)
        const uchar* prevPtr = img.data + img.step*ymap[y-1];
        const uchar* nextPtr = img.data + img.step*ymap[y+1];

        float* gradPtr = grad.ptr<float>(y);
        uchar* qanglePtr = qangle.ptr(y);

        if( cn == 1 )
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x];
                dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
                dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
            }
        }
        else
        {
            x = 0;
#if CV_SSE2
            for( ; x <= width - 4; x += 4 )
            {
                int x0 = xmap[x], x1 = xmap[x+1], x2 = xmap[x+2], x3 = xmap[x+3];
                typedef const uchar* const T;
                T p02 = imgPtr + xmap[x+1], p00 = imgPtr + xmap[x-1];
                T p12 = imgPtr + xmap[x+2], p10 = imgPtr + xmap[x];
                T p22 = imgPtr + xmap[x+3], p20 = p02;
                T p32 = imgPtr + xmap[x+4], p30 = p12;

                __m128 _dx0 = _mm_sub_ps(_mm_set_ps(lut[p32[0]], lut[p22[0]], lut[p12[0]], lut[p02[0]]),
                                         _mm_set_ps(lut[p30[0]], lut[p20[0]], lut[p10[0]], lut[p00[0]]));
                __m128 _dx1 = _mm_sub_ps(_mm_set_ps(lut[p32[1]], lut[p22[1]], lut[p12[1]], lut[p02[1]]),
                                         _mm_set_ps(lut[p30[1]], lut[p20[1]], lut[p10[1]], lut[p00[1]]));
                __m128 _dx2 = _mm_sub_ps(_mm_set_ps(lut[p32[2]], lut[p22[2]], lut[p12[2]], lut[p02[2]]),
                                         _mm_set_ps(lut[p30[2]], lut[p20[2]], lut[p10[2]], lut[p00[2]]));

                __m128 _dy0 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3]], lut[nextPtr[x2]], lut[nextPtr[x1]], lut[nextPtr[x0]]),
                                         _mm_set_ps(lut[prevPtr[x3]], lut[prevPtr[x2]], lut[prevPtr[x1]], lut[prevPtr[x0]]));
                __m128 _dy1 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3+1]], lut[nextPtr[x2+1]], lut[nextPtr[x1+1]], lut[nextPtr[x0+1]]),
                                         _mm_set_ps(lut[prevPtr[x3+1]], lut[prevPtr[x2+1]], lut[prevPtr[x1+1]], lut[prevPtr[x0+1]]));
                __m128 _dy2 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3+2]], lut[nextPtr[x2+2]], lut[nextPtr[x1+2]], lut[nextPtr[x0+2]]),
                                         _mm_set_ps(lut[prevPtr[x3+2]], lut[prevPtr[x2+2]], lut[prevPtr[x1+2]], lut[prevPtr[x0+2]]));

                __m128 _mag0 = _mm_add_ps(_mm_mul_ps(_dx0, _dx0), _mm_mul_ps(_dy0, _dy0));
                __m128 _mag1 = _mm_add_ps(_mm_mul_ps(_dx1, _dx1), _mm_mul_ps(_dy1, _dy1));
                __m128 _mag2 = _mm_add_ps(_mm_mul_ps(_dx2, _dx2), _mm_mul_ps(_dy2, _dy2));

                __m128 mask = _mm_cmpgt_ps(_mag2, _mag1);
                _dx2 = _mm_or_ps(_mm_and_ps(_dx2, mask), _mm_andnot_ps(mask, _dx1));
                _dy2 = _mm_or_ps(_mm_and_ps(_dy2, mask), _mm_andnot_ps(mask, _dy1));

                mask = _mm_cmpgt_ps(_mm_max_ps(_mag2, _mag1), _mag0);
                _dx2 = _mm_or_ps(_mm_and_ps(_dx2, mask), _mm_andnot_ps(mask, _dx0));
                _dy2 = _mm_or_ps(_mm_and_ps(_dy2, mask), _mm_andnot_ps(mask, _dy0));

                _mm_storeu_ps(dbuf + x, _dx2);
                _mm_storeu_ps(dbuf + x + width, _dy2);
            }
#elif CV_NEON
            for( ; x <= width - 4; x += 4 )
            {
                int x0 = xmap[x], x1 = xmap[x+1], x2 = xmap[x+2], x3 = xmap[x+3];
                typedef const uchar* const T;
                T p02 = imgPtr + xmap[x+1], p00 = imgPtr + xmap[x-1];
                T p12 = imgPtr + xmap[x+2], p10 = imgPtr + xmap[x];
                T p22 = imgPtr + xmap[x+3], p20 = p02;
                T p32 = imgPtr + xmap[x+4], p30 = p12;

                float32x4_t _dx0 = vsubq_f32(vsetq_f32(lut[p02[0]], lut[p12[0]], lut[p22[0]], lut[p32[0]]),
                                             vsetq_f32(lut[p00[0]], lut[p10[0]], lut[p20[0]], lut[p30[0]]));
                float32x4_t _dx1 = vsubq_f32(vsetq_f32(lut[p02[1]], lut[p12[1]], lut[p22[1]], lut[p32[1]]),
                                             vsetq_f32(lut[p00[1]], lut[p10[1]], lut[p20[1]], lut[p30[1]]));
                float32x4_t _dx2 = vsubq_f32(vsetq_f32(lut[p02[2]], lut[p12[2]], lut[p22[2]], lut[p32[2]]),
                                             vsetq_f32(lut[p00[2]], lut[p10[2]], lut[p20[2]], lut[p30[2]]));

                float32x4_t _dy0 = vsubq_f32(vsetq_f32(lut[nextPtr[x0]], lut[nextPtr[x1]], lut[nextPtr[x2]], lut[nextPtr[x3]]),
                                             vsetq_f32(lut[prevPtr[x0]], lut[prevPtr[x1]], lut[prevPtr[x2]], lut[prevPtr[x3]]));
                float32x4_t _dy1 = vsubq_f32(vsetq_f32(lut[nextPtr[x0+1]], lut[nextPtr[x1+1]], lut[nextPtr[x2+1]], lut[nextPtr[x3+1]]),
                                             vsetq_f32(lut[prevPtr[x0+1]], lut[prevPtr[x1+1]], lut[prevPtr[x2+1]], lut[prevPtr[x3+1]]));
                float32x4_t _dy2 = vsubq_f32(vsetq_f32(lut[nextPtr[x0+2]], lut[nextPtr[x1+2]], lut[nextPtr[x2+2]], lut[nextPtr[x3+2]]),
                                             vsetq_f32(lut[prevPtr[x0+2]], lut[prevPtr[x1+2]], lut[prevPtr[x2+2]], lut[prevPtr[x3+2]]));

                float32x4_t _mag0 = vaddq_f32(vmulq_f32(_dx0, _dx0), vmulq_f32(_dy0, _dy0));
                float32x4_t _mag1 = vaddq_f32(vmulq_f32(_dx1, _dx1), vmulq_f32(_dy1, _dy1));
                float32x4_t _mag2 = vaddq_f32(vmulq_f32(_dx2, _dx2), vmulq_f32(_dy2, _dy2));

                uint32x4_t mask = vcgtq_f32(_mag2, _mag1);
                _dx2 = vbslq_f32(mask, _dx2, _dx1);
                _dy2 = vbslq_f32(mask, _dy2, _dy1);

                mask = vcgtq_f32(vmaxq_f32(_mag2, _mag1), _mag0);
                _dx2 = vbslq_f32(mask, _dx2, _dx0);
                _dy2 = vbslq_f32(mask, _dy2, _dy0);

                vst1q_f32(dbuf + x, _dx2);
                vst1q_f32(dbuf + x + width, _dy2);
            }
#endif
            for( ; x < width; x++ )
            {
                int x1 = xmap[x];
                float dx0, dy0, dx, dy, mag0, mag;
                const uchar* p2 = imgPtr + xmap[x+1];
                const uchar* p0 = imgPtr + xmap[x-1];

                dx0 = lut[p2[2]] - lut[p0[2]];
                dy0 = lut[nextPtr[x1+2]] - lut[prevPtr[x1+2]];
                mag0 = dx0*dx0 + dy0*dy0;

                dx = lut[p2[1]] - lut[p0[1]];
                dy = lut[nextPtr[x1+1]] - lut[prevPtr[x1+1]];
                mag = dx*dx + dy*dy;
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dx = lut[p2[0]] - lut[p0[0]];
                dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
                mag = dx*dx + dy*dy;
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dbuf[x] = dx0;
                dbuf[x+width] = dy0;
            }
        }

        // computing angles and magnidutes
        cartToPolar( Dx, Dy, Mag, Angle, false );

        // filling the result matrix
        x = 0;
#if CV_SSE2
        __m128 fhalf = _mm_set1_ps(0.5f), fzero = _mm_setzero_ps();
        __m128 _angleScale = _mm_set1_ps(angleScale), fone = _mm_set1_ps(1.0f);
        __m128i ione = _mm_set1_epi32(1), _nbins = _mm_set1_epi32(nbins), izero = _mm_setzero_si128();

        for ( ; x <= width - 4; x += 4)
        {
            int x2 = x << 1;
            __m128 _mag = _mm_loadu_ps(dbuf + x + (width << 1));
            __m128 _angle = _mm_loadu_ps(dbuf + x + width * 3);
            _angle = _mm_sub_ps(_mm_mul_ps(_angleScale, _angle), fhalf);

            __m128 sign = _mm_and_ps(fone, _mm_cmplt_ps(_angle, fzero));
            __m128i _hidx = _mm_cvttps_epi32(_angle);
            _hidx = _mm_sub_epi32(_hidx, _mm_cvtps_epi32(sign));
