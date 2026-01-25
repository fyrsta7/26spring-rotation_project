    bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const
    {
        const double outlierRatio = 0.45;
        bool result = false;
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        Mat ms1, ms2, err, errf, model, bestModel, mask, mask0;

        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
        double minMedian = DBL_MAX;

        RNG rng((uint64)-1);

        CV_Assert( cb );
        CV_Assert( confidence > 0 && confidence < 1 );

        CV_Assert( count >= 0 && count2 == count );
        if( count < modelPoints )
            return false;

        if( _mask.needed() )
        {
            _mask.create(count, 1, CV_8U, -1, true);
            mask0 = mask = _mask.getMat();
            CV_Assert( (mask.cols == 1 || mask.rows == 1) && (int)mask.total() == count );
        }

        if( count == modelPoints )
        {
            if( cb->runKernel(m1, m2, bestModel) <= 0 )
                return false;
            bestModel.copyTo(_model);
            mask.setTo(Scalar::all(1));
            return true;
        }

        int iter, niters = RANSACUpdateNumIters(confidence, outlierRatio, modelPoints, maxIters);
        niters = MAX(niters, 3);

        for( iter = 0; iter < niters; iter++ )
        {
            int i, nmodels;
            if( count > modelPoints )
            {
                bool found = getSubset( m1, m2, ms1, ms2, rng );
                if( !found )
                {
                    if( iter == 0 )
                        return false;
                    break;
                }
            }

            nmodels = cb->runKernel( ms1, ms2, model );
            if( nmodels <= 0 )
                continue;

            CV_Assert( model.rows % nmodels == 0 );
            Size modelSize(model.cols, model.rows/nmodels);

            for( i = 0; i < nmodels; i++ )
            {
                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
                cb->computeError( m1, m2, model_i, err );
                if( err.depth() != CV_32F )
                    err.convertTo(errf, CV_32F);
                else
                    errf = err;
                CV_Assert( errf.isContinuous() && errf.type() == CV_32F && (int)errf.total() == count );
                std::sort(errf.ptr<int>(), errf.ptr<int>() + count);

                double median = count % 2 != 0 ?
                errf.at<float>(count/2) : (errf.at<float>(count/2-1) + errf.at<float>(count/2))*0.5;

                if( median < minMedian )
                {
                    minMedian = median;
                    model_i.copyTo(bestModel);
                }
            }
        }

        if( minMedian < DBL_MAX )
        {
            double sigma = 2.5*1.4826*(1 + 5./(count - modelPoints))*std::sqrt(minMedian);
            sigma = MAX( sigma, 0.001 );

            count = findInliers( m1, m2, bestModel, err, mask, sigma );
            if( _mask.needed() && mask0.data != mask.data )
            {
                if( mask0.size() == mask.size() )
                    mask.copyTo(mask0);
                else
                    transpose(mask, mask0);
            }
            bestModel.copyTo(_model);
            result = count >= modelPoints;
        }
        else
            _model.release();

        return result;
    }