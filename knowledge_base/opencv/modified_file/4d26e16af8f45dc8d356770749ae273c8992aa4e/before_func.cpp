CNode& ContourScanner_::makeContour(schar& nbd_, const bool is_hole, const int x, const int y)
{
    const bool isChain = (this->approx_method1 == CV_CHAIN_CODE);  // TODO: get rid of old constant
    const bool isDirect = (this->approx_method1 == CHAIN_APPROX_NONE);

    const Point start_pt(x - (is_hole ? 1 : 0), y);

    CNode& res = tree.newElem();
    if (isChain)
        res.body.codes.reserve(200);
    else
        res.body.pts.reserve(200);
    res.body.isHole = is_hole;
    res.body.isChain = isChain;
    res.body.origin = start_pt + offset;
    if (isSimple())
    {
        icvFetchContourEx<schar>(this->image, start_pt, MASK8_NEW, res.body, isDirect);
    }
    else
    {
        schar lval;
        if (isInt())
        {
            const int start_val = this->image.at<int>(start_pt);
            lval = start_val & MASK8_LVAL;
            icvFetchContourEx<int>(this->image, start_pt, 0, res.body, isDirect);
        }
        else
        {
            lval = nbd_;
            // change nbd
            nbd_ = (nbd_ + 1) & MASK8_LVAL;
            if (nbd_ == 0)
                nbd_ = MASK8_BLACK | MASK8_NEW;
            icvFetchContourEx<schar>(this->image, start_pt, lval, res.body, isDirect);
        }
        res.body.brect.x -= this->offset.x;
        res.body.brect.y -= this->offset.y;
        res.ctable_next = this->ctable[lval];
        this->ctable[lval] = res.self();
    }
    const Point prev_origin = res.body.origin;
    res.body.origin = start_pt;
    if (this->approx_method1 != this->approx_method2)
    {
        CV_Assert(res.body.isChain);
        res.body.pts = approximateChainTC89(res.body.codes, prev_origin, this->approx_method2);
        res.body.isChain = false;
    }
    return res;
}
