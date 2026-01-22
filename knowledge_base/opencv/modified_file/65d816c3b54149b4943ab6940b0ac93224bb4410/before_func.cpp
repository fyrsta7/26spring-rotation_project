    virtual void doTrain(InputArray points) { (void)points; }

    void clear()
    {
        samples.release();
        responses.release();
    }

    void read( const FileNode& fn )
    {
        clear();
        isclassifier = (int)fn["is_classifier"] != 0;
        defaultK = (int)fn["default_k"];

        fn["samples"] >> samples;
        fn["responses"] >> responses;
    }

    void write( FileStorage& fs ) const
    {
        fs << "is_classifier" << (int)isclassifier;
        fs << "default_k" << defaultK;

        fs << "samples" << samples;
        fs << "responses" << responses;
    }

public:
    int defaultK;
    bool isclassifier;
    int Emax;

    Mat samples;
    Mat responses;
};

class BruteForceImpl CV_FINAL : public Impl
{
public:
    String getModelName() const CV_OVERRIDE { return NAME_BRUTE_FORCE; }
    int getType() const CV_OVERRIDE { return ml::KNearest::BRUTE_FORCE; }

    void findNearestCore( const Mat& _samples, int k0, const Range& range,
                          Mat* results, Mat* neighbor_responses,
                          Mat* dists, float* presult ) const
    {
        int testidx, baseidx, i, j, d = samples.cols, nsamples = samples.rows;
        int testcount = range.end - range.start;
        int k = std::min(k0, nsamples);

        AutoBuffer<float> buf(testcount*k*2);
        float* dbuf = buf;
        float* rbuf = dbuf + testcount*k;

        const float* rptr = responses.ptr<float>();

        for( testidx = 0; testidx < testcount; testidx++ )
        {
            for( i = 0; i < k; i++ )
            {
                dbuf[testidx*k + i] = FLT_MAX;
                rbuf[testidx*k + i] = 0.f;
            }
        }

        for( baseidx = 0; baseidx < nsamples; baseidx++ )
        {
            for( testidx = 0; testidx < testcount; testidx++ )
            {
                const float* v = samples.ptr<float>(baseidx);
                const float* u = _samples.ptr<float>(testidx + range.start);

                float s = 0;
                for( i = 0; i <= d - 4; i += 4 )
                {
                    float t0 = u[i] - v[i], t1 = u[i+1] - v[i+1];
                    float t2 = u[i+2] - v[i+2], t3 = u[i+3] - v[i+3];
                    s += t0*t0 + t1*t1 + t2*t2 + t3*t3;
                }

                for( ; i < d; i++ )
                {
                    float t0 = u[i] - v[i];
                    s += t0*t0;
                }

                Cv32suf si;
                si.f = (float)s;
                Cv32suf* dd = (Cv32suf*)(&dbuf[testidx*k]);
                float* nr = &rbuf[testidx*k];

                for( i = k; i > 0; i-- )
                    if( si.i >= dd[i-1].i )
                        break;
                if( i >= k )
                    continue;

                for( j = k-2; j >= i; j-- )
                {
                    dd[j+1].i = dd[j].i;
                    nr[j+1] = nr[j];
                }
                dd[i].i = si.i;
                nr[i] = rptr[baseidx];
            }
        }

        float result = 0.f;
        float inv_scale = 1.f/k;

        for( testidx = 0; testidx < testcount; testidx++ )
        {
            if( neighbor_responses )
            {
                float* nr = neighbor_responses->ptr<float>(testidx + range.start);
                for( j = 0; j < k; j++ )
                    nr[j] = rbuf[testidx*k + j];
                for( ; j < k0; j++ )
                    nr[j] = 0.f;
            }

            if( dists )
            {
                float* dptr = dists->ptr<float>(testidx + range.start);
                for( j = 0; j < k; j++ )
                    dptr[j] = dbuf[testidx*k + j];
                for( ; j < k0; j++ )
                    dptr[j] = 0.f;
            }

            if( results || testidx+range.start == 0 )
            {
                if( !isclassifier || k == 1 )
                {
                    float s = 0.f;
                    for( j = 0; j < k; j++ )
                        s += rbuf[testidx*k + j];
                    result = (float)(s*inv_scale);
                }
