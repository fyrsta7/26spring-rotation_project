      { .str = "0" }, .flags = FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(hue);

static inline void compute_sin_and_cos(HueContext *hue)
{
    /*
     * Scale the value to the norm of the resulting (U,V) vector, that is
     * the saturation.
     * This will be useful in the apply_lut function.
     */
    hue->hue_sin = rint(sin(hue->hue) * (1 << 16) * hue->saturation);
    hue->hue_cos = rint(cos(hue->hue) * (1 << 16) * hue->saturation);
}

static inline void create_luma_lut(HueContext *h)
{
    const float b = h->brightness;
    int i;
