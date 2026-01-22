                iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                float errval = 0.f;
                
                for( y = 0; y < winSize.height; y++ )
                {
                    const uchar* Jptr = (const uchar*)J.data + (y + inextPt.y)*step + inextPt.x*cn;
                    const deriv_type* Iptr = (const deriv_type*)(IWinBuf.data + y*IWinBuf.step);
                    
                    for( x = 0; x < winSize.width*cn; x++ )
                    {
                        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                              Jptr[x+step]*iw10 + Jptr[x+step+cn]*iw11,
                                              W_BITS1-5) - Iptr[x];
                        errval += std::abs((float)diff);
                    }
                }
                err[ptidx] = errval * 1.f/(32*winSize.width*cn*winSize.height);
            }
        }
    }
    
    const Mat* prevImg;
    const Mat* nextImg;
    const Mat* prevDeriv;
    const Point2f* prevPts;
    Point2f* nextPts;
    uchar* status;
    float* err;
    Size winSize;
    TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;
};
    
}

void cv::calcOpticalFlowPyrLK( InputArray _prevImg, InputArray _nextImg,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err,
                           Size winSize, int maxLevel,
                           TermCriteria criteria,
                           double derivLambda,
                           int flags, double minEigThreshold )
{
#ifdef HAVE_TEGRA_OPTIMIZATION__DISABLED
    if (tegra::calcOpticalFlowPyrLK(_prevImg, _nextImg, _prevPts, _nextPts, _status, _err, winSize, maxLevel, criteria, derivLambda, flags))
        return;
#endif
    Mat prevImg = _prevImg.getMat(), nextImg = _nextImg.getMat(), prevPtsMat = _prevPts.getMat();
    derivLambda = std::min(std::max(derivLambda, 0.), 1.);
    const int derivDepth = DataType<deriv_type>::depth;

    CV_Assert( derivLambda >= 0 );
    CV_Assert( maxLevel >= 0 && winSize.width > 2 && winSize.height > 2 );
    CV_Assert( prevImg.size() == nextImg.size() &&
        prevImg.type() == nextImg.type() );

    int level=0, i, k, npoints, cn = prevImg.channels(), cn2 = cn*2;
    CV_Assert( (npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0 );
    
    if( npoints == 0 )
    {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }
    
    if( !(flags & OPTFLOW_USE_INITIAL_FLOW) )
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);
    
    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert( nextPtsMat.checkVector(2, CV_32F, true) == npoints );
    
    const Point2f* prevPts = (const Point2f*)prevPtsMat.data;
    Point2f* nextPts = (Point2f*)nextPtsMat.data;
    
    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.data;
    float* err = 0;
    
    for( i = 0; i < npoints; i++ )
        status[i] = true;
    
    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = (float*)errMat.data;
    }

    vector<Mat> prevPyr(maxLevel+1), nextPyr(maxLevel+1);
    
    // build the image pyramids.
    // we pad each level with +/-winSize.{width|height}
    // pixels to simplify the further patch extraction.
    // Thanks to the reference counting, "temp" mat (the pyramid layer + border)
    // will not be deallocated, since {prevPyr|nextPyr}[level] will be a ROI in "temp".
    for( k = 0; k < 2; k++ )
    {
        Size sz = prevImg.size();
        vector<Mat>& pyr = k == 0 ? prevPyr : nextPyr;
        Mat& img0 = k == 0 ? prevImg : nextImg;
        
        for( level = 0; level <= maxLevel; level++ )
        {
            Mat temp(sz.height + winSize.height*2,
                     sz.width + winSize.width*2,
                     img0.type());
            pyr[level] = temp(Rect(winSize.width, winSize.height, sz.width, sz.height));
            if( level == 0 )
                img0.copyTo(pyr[level]);
            else
                pyrDown(pyr[level-1], pyr[level], pyr[level].size());
            copyMakeBorder(pyr[level], temp, winSize.height, winSize.height,
                           winSize.width, winSize.width, BORDER_REFLECT_101|BORDER_ISOLATED);
