          firstOctave(_firstOctave) { }

    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
        const int begin = range.start;
        const int end = range.end;

        static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

        for ( int i = begin; i<end; i++ )
        {
            KeyPoint kpt = keypoints[i];
            int octave, layer;
            float scale;
            unpackOctave(kpt, octave, layer, scale);
            CV_Assert(octave >= firstOctave && layer <= nOctaveLayers+2);
            float size=kpt.size*scale;
            Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
            const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

            float angle = 360.f - kpt.angle;
            if(std::abs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;
            calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
        }
    }
private:
    const std::vector<Mat>& gpyr;
    const std::vector<KeyPoint>& keypoints;
    Mat& descriptors;
    int nOctaveLayers;
    int firstOctave;
};

static void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
                            Mat& descriptors, int nOctaveLayers, int firstOctave )
{
    parallel_for_(Range(0, static_cast<int>(keypoints.size())), calcDescriptorsComputer(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave));
}

//////////////////////////////////////////////////////////////////////////////////////////

SIFT_Impl::SIFT_Impl( int _nfeatures, int _nOctaveLayers,
           double _contrastThreshold, double _edgeThreshold, double _sigma )
    : nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
    contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
{
}

int SIFT_Impl::descriptorSize() const
{
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int SIFT_Impl::descriptorType() const
{
    return CV_32F;
}

int SIFT_Impl::defaultNorm() const
{
    return NORM_L2;
}


void SIFT_Impl::detectAndCompute(InputArray _image, InputArray _mask,
                      std::vector<KeyPoint>& keypoints,
                      OutputArray _descriptors,
                      bool useProvidedKeypoints)
{
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    Mat image = _image.getMat(), mask = _mask.getMat();

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( !mask.empty() && mask.type() != CV_8UC1 )
        CV_Error( Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

    if( useProvidedKeypoints )
    {
        firstOctave = 0;
        int maxOctave = INT_MIN;
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            int octave, layer;
            float scale;
