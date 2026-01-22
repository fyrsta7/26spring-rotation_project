
// Does multi-dimensional linear interpolation over a lookup table.
// `size(i)` should returns the number of samples in the i'th dimension.
// `sample()` gets a vector where 0 <= i'th coordinate < size(i) and should return the value of the look-up table at that position.
inline FloatVector3 lerp_nd(Function<unsigned(size_t)> size, Function<FloatVector3(ReadonlySpan<unsigned> const&)> sample, Vector<float> const& x)
{
    unsigned left_index[x.size()];
    float factor[x.size()];
    for (size_t i = 0; i < x.size(); ++i) {
        unsigned n = size(i) - 1;
        float ec = x[i] * n;
        left_index[i] = min(static_cast<unsigned>(ec), n - 1);
        factor[i] = ec - left_index[i];
    }

    FloatVector3 sample_output {};
    // The i'th bit of mask indicates if the i'th coordinate is rounded up or down.
    Vector<unsigned> coordinates;
    coordinates.resize(x.size());
    for (size_t mask = 0; mask < (1u << x.size()); ++mask) {
        float sample_weight = 1.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            coordinates[i] = left_index[i] + ((mask >> i) & 1u);
            sample_weight *= ((mask >> i) & 1u) ? factor[i] : 1.0f - factor[i];
        }
        sample_output += sample(coordinates) * sample_weight;
