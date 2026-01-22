        {
            iIdxBGMax++;
            iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];
        }

        for (unsigned n = iIdxBGMax + 1; n < iCntMaxima; n++)
        {
            if (piHistIntensity[piMaxPos[n]] >= iMaxVal)
            {
                iMaxVal = piHistIntensity[piMaxPos[n]];
                iIdxBGMax = n;
            }
        }

        //SETTING THRESHOLD FOR BINARIZATION
        int iDist2 = (iBrightMax - piMaxPos[iIdxBGMax])/2;
        iThresh = iBrightMax - iDist2;
        DPRINTF("THRESHOLD SELECTED = %d, BRIGHTMAX = %d, DARKMAX = %d", iThresh, iBrightMax, piMaxPos[iIdxBGMax]);
    }

    if (iThresh > 0)
    {
        img = (img >= iThresh);
    }
}

bool findChessboardCorners(InputArray image_, Size pattern_size,
                           OutputArray corners_, int flags)
{
    CV_INSTRUMENT_REGION();

    DPRINTF("==== findChessboardCorners(img=%dx%d, pattern=%dx%d, flags=%d)",
            image_.cols(), image_.rows(), pattern_size.width, pattern_size.height, flags);

    bool found = false;

    const bool is_plain = (flags & CALIB_CB_PLAIN) != 0;

    int type = image_.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    Mat img = image_.getMat();

    CV_CheckType(type, depth == CV_8U && (cn == 1 || cn == 3 || cn == 4),
            "Only 8-bit grayscale or color images are supported");

    if (pattern_size.width <= 2 || pattern_size.height <= 2)
        CV_Error(Error::StsOutOfRange, "Both width and height of the pattern should have bigger than 2");

    if (!corners_.needed())
        CV_Error(Error::StsNullPtr, "Null pointer to corners");

    std::vector<cv::Point2f> out_corners;

    if (is_plain)
      CV_CheckType(type, depth == CV_8U && cn == 1, "Only 8-bit grayscale images are supported whith CALIB_CB_PLAIN flag enable");

    if (img.channels() != 1)
    {
        cvtColor(img, img, COLOR_BGR2GRAY);
    }

    int prev_sqr_size = 0;

    Mat thresh_img_new = img.clone();
    if(!is_plain)
        icvBinarizationHistogramBased(thresh_img_new); // process image in-place
    SHOW("New binarization", thresh_img_new);

    if (flags & CALIB_CB_FAST_CHECK && !is_plain)
    {
        //perform new method for checking chessboard using a binary image.
        //image is binarised using a threshold dependent on the image histogram
        if (checkChessboardBinary(thresh_img_new, pattern_size) <= 0) //fall back to the old method
        {
            if (!checkChessboard(img, pattern_size))
            {
                corners_.release();
                return false;
            }
        }
    }

    ChessBoardDetector detector(pattern_size);

    const int min_dilations = 0;
    const int max_dilations = is_plain ? 0 : 7;

    // Try our standard "0" and "1" dilations, but if the pattern is not found, iterate the whole procedure with higher dilations.
    // This is necessary because some squares simply do not separate properly without and with a single dilations. However,
    // we want to use the minimum number of dilations possible since dilations cause the squares to become smaller,
    // making it difficult to detect smaller squares.
    for (int dilations = min_dilations; dilations <= max_dilations; dilations++)
    {
        //USE BINARY IMAGE COMPUTED USING icvBinarizationHistogramBased METHOD
        if(!is_plain && dilations > 0)
            dilate( thresh_img_new, thresh_img_new, Mat(), Point(-1, -1), 1 );

        // So we can find rectangles that go to the edge, we draw a white line around the image edge.
        // Otherwise FindContours will miss those clipped rectangle contours.
        // The border color will be the image mean, because otherwise we risk screwing up filters like cvSmooth()...
        rectangle( thresh_img_new, Point(0,0), Point(thresh_img_new.cols-1, thresh_img_new.rows-1), Scalar(255,255,255), 3, LINE_8);

        detector.reset();
        detector.generateQuads(thresh_img_new, flags, dilations);
        DPRINTF("Quad count: %d/%d", detector.all_quads_count, (pattern_size.width/2+1)*(pattern_size.height/2+1));
        SHOW_QUADS("New quads", thresh_img_new, &detector.all_quads[0], detector.all_quads_count);
        if (detector.processQuads(out_corners, prev_sqr_size))
        {
            found = true;
            break;
        }
    }

    DPRINTF("Chessboard detection result 0: %d", (int)found);

    // revert to old, slower, method if detection failed
    if (!found && !is_plain)
    {
        if (flags & CALIB_CB_NORMALIZE_IMAGE)
        {
            img = img.clone();
            equalizeHist(img, img);
        }

        Mat thresh_img;
        prev_sqr_size = 0;

        DPRINTF("Fallback to old algorithm");
        const bool useAdaptive = flags & CALIB_CB_ADAPTIVE_THRESH;
        if (!useAdaptive)
        {
            // empiric threshold level
            // thresholding performed here and not inside the cycle to save processing time
            double mean = cv::mean(img).val[0];
            int thresh_level = std::max(cvRound(mean - 10), 10);
            threshold(img, thresh_img, thresh_level, 255, THRESH_BINARY);
        }
        //if flag CALIB_CB_ADAPTIVE_THRESH is not set it doesn't make sense to iterate over k
        int max_k = useAdaptive ? 6 : 1;
        Mat prev_thresh_img;
        for (int k = 0; k < max_k && !found; k++)
        {
            int prev_block_size = -1;
            for (int dilations = min_dilations; dilations <= max_dilations; dilations++)
            {
                // convert the input grayscale image to binary (black-n-white)
                if (useAdaptive)
                {
                    int block_size = cvRound(prev_sqr_size == 0
                                             ? std::min(img.cols, img.rows) * (k % 2 == 0 ? 0.2 : 0.1)
                                             : prev_sqr_size * 2);
                    block_size = block_size | 1;
                    // convert to binary
                    if (block_size != prev_block_size)
                    {
                        adaptiveThreshold( img, thresh_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size, (k/2)*5 );
                        dilate( thresh_img, thresh_img, Mat(), Point(-1, -1), dilations );
                        thresh_img.copyTo(prev_thresh_img);
                    }
                    else if (dilations > 0)
                    {
                        dilate( prev_thresh_img, prev_thresh_img, Mat(), Point(-1, -1), 1 );
                        prev_thresh_img.copyTo(thresh_img);
                    }
                    prev_block_size = block_size;
                }
                else
                {
                    if (dilations > 0)
                        dilate( thresh_img, thresh_img, Mat(), Point(-1, -1), 1 );
                }
                SHOW("Old binarization", thresh_img);

                // So we can find rectangles that go to the edge, we draw a white line around the image edge.
                // Otherwise FindContours will miss those clipped rectangle contours.
                // The border color will be the image mean, because otherwise we risk screwing up filters like cvSmooth()...
                rectangle( thresh_img, Point(0,0), Point(thresh_img.cols-1, thresh_img.rows-1), Scalar(255,255,255), 3, LINE_8);

                detector.reset();
                detector.generateQuads(thresh_img, flags, dilations);
                DPRINTF("Quad count: %d/%d", detector.all_quads_count, (pattern_size.width/2+1)*(pattern_size.height/2+1));
                SHOW_QUADS("Old quads", thresh_img, &detector.all_quads[0], detector.all_quads_count);
                if (detector.processQuads(out_corners, prev_sqr_size))
                {
                    found = 1;
                    break;
                }
            }
        }
    }

    DPRINTF("Chessboard detection result 1: %d", (int)found);

    if (found)
        found = detector.checkBoardMonotony(out_corners);

    DPRINTF("Chessboard detection result 2: %d", (int)found);

    // check that none of the found corners is too close to the image boundary
    if (found)
    {
        const int BORDER = 8;
        for (int k = 0; k < pattern_size.width*pattern_size.height; ++k)
        {
            if( out_corners[k].x <= BORDER || out_corners[k].x > img.cols - BORDER ||
                out_corners[k].y <= BORDER || out_corners[k].y > img.rows - BORDER )
            {
                found = false;
                break;
            }
        }
