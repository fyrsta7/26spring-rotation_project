}

float normL2Sqr_(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
#if CV_SIMD
    v_float32 v_d0 = vx_setzero_f32(), v_d1 = vx_setzero_f32();
    v_float32 v_d2 = vx_setzero_f32(), v_d3 = vx_setzero_f32();
    for (; j <= n - 4 * v_float32::nlanes; j += 4 * v_float32::nlanes)
    {
        v_float32 t0 = vx_load(a + j) - vx_load(b + j);
        v_float32 t1 = vx_load(a + j + v_float32::nlanes) - vx_load(b + j + v_float32::nlanes);
        v_float32 t2 = vx_load(a + j + 2 * v_float32::nlanes) - vx_load(b + j + 2 * v_float32::nlanes);
        v_float32 t3 = vx_load(a + j + 3 * v_float32::nlanes) - vx_load(b + j + 3 * v_float32::nlanes);
        v_d0 = v_muladd(t0, t0, v_d0);
        v_d1 = v_muladd(t1, t1, v_d1);
        v_d2 = v_muladd(t2, t2, v_d2);
        v_d3 = v_muladd(t3, t3, v_d3);
    }
    d = v_reduce_sum(v_d0 + v_d1 + v_d2 + v_d3);
#endif
    for( ; j < n; j++ )
    {
        float t = a[j] - b[j];
        d += t*t;
    }
