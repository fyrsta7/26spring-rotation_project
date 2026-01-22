static String oclGetTMacro(const UMat &m)
{
    String str_name = ocl::typeToStr(m.type());

    if (str_name == "short")
        str_name = "half";

    return format("-DT=%s -Dconvert_T=convert_%s ", str_name.c_str(), str_name.c_str());
}
#endif

struct BaseFunctor
{
    void finalize() {}

    bool tryFuse(Ptr<dnn::Layer>&) { return false; }

    void getScaleShift(Mat&, Mat&) const {}
};

struct ReLUFunctor : public BaseFunctor
{
    typedef ReLULayer Layer;
    float slope;

    explicit ReLUFunctor(float slope_=1.f) : slope(slope_) {}

    bool supportBackend(int backendId, int)
    {
#ifdef HAVE_DNN_IE_NN_BUILDER_2019
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
