
    nframes = 0;
    history = defaultHistory2;
    varThreshold = defaultVarThreshold2;
    bShadowDetection = 1;

    nmixtures = defaultNMixtures2;
    backgroundRatio = defaultBackgroundRatio2;
    fVarInit = defaultVarInit2;
    fVarMax  = defaultVarMax2;
    fVarMin = defaultVarMin2;

    varThresholdGen = defaultVarThresholdGen2;
    fCT = defaultfCT2;
    nShadowDetection =  defaultnShadowDetection2;
    fTau = defaultfTau;
}

BackgroundSubtractorMOG2::BackgroundSubtractorMOG2(int _history,  float _varThreshold, bool _bShadowDetection)
{
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    history = _history > 0 ? _history : defaultHistory2;
    varThreshold = (_varThreshold>0)? _varThreshold : defaultVarThreshold2;
    bShadowDetection = _bShadowDetection;

    nmixtures = defaultNMixtures2;
    backgroundRatio = defaultBackgroundRatio2;
    fVarInit = defaultVarInit2;
    fVarMax  = defaultVarMax2;
    fVarMin = defaultVarMin2;

    varThresholdGen = defaultVarThresholdGen2;
    fCT = defaultfCT2;
    nShadowDetection =  defaultnShadowDetection2;
    fTau = defaultfTau;
}

BackgroundSubtractorMOG2::~BackgroundSubtractorMOG2()
{
}


void BackgroundSubtractorMOG2::initialize(Size _frameSize, int _frameType)
