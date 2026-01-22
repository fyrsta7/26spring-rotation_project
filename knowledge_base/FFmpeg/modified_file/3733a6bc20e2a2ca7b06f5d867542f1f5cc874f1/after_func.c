
    normalize_vector(vec);

    return 1;
}

/**
 * Calculate frame position in hammer format for corresponding 3D coordinates on sphere.
 *
 * @param s filter private context
 * @param vec coordinates on sphere
 * @param width frame width
 * @param height frame height
 * @param us horizontal coordinates for interpolation window
 * @param vs vertical coordinates for interpolation window
 * @param du horizontal relative coordinate
 * @param dv vertical relative coordinate
 */
static int xyz_to_hammer(const V360Context *s,
                         const float *vec, int width, int height,
                         int16_t us[4][4], int16_t vs[4][4], float *du, float *dv)
{
    const float theta = atan2f(vec[0], -vec[2]) * s->input_mirror_modifier[0];

    const float z = sqrtf(1.f + sqrtf(1.f - vec[1] * vec[1]) * cosf(theta * 0.5f));
    const float x = sqrtf(1.f - vec[1] * vec[1]) * sinf(theta * 0.5f) / z;
    const float y = -vec[1] / z * s->input_mirror_modifier[1];
    float uf, vf;
    int ui, vi;

    uf = (x + 1.f) * width  / 2.f;
    vf = (y + 1.f) * height / 2.f;
