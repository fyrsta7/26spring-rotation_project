 * @param height frame height
 * @param vec coordinates on sphere
 */
static int ball_to_xyz(const V360Context *s,
                       int i, int j, int width, int height,
                       float *vec)
{
    const float x = (2.f * i + 1.f) / width  - 1.f;
    const float y = (2.f * j + 1.f) / height - 1.f;
    const float l = hypotf(x, y);

    if (l <= 1.f) {
        const float z = 2.f * l * sqrtf(1.f - l * l);

        vec[0] = z * x / (l > 0.f ? l : 1.f);
        vec[1] = z * y / (l > 0.f ? l : 1.f);
        vec[2] = 1.f - 2.f * l * l;
    } else {
