static int countNonZero64f( const double* src, int len )
{
    int nz = 0, i = 0;
#if CV_SIMD_64F
    v_int64 sum1 = vx_setzero_s64();
    v_int64 sum2 = vx_setzero_s64();
    v_float64 zero = vx_setzero_f64();
    int step = v_float64::nlanes * 2;
    int len0 = len & -step;

    for(i = 0; i < len0; i += step )
        {
        sum1 += v_reinterpret_as_s64(vx_load(&src[i]) == zero);
        sum2 += v_reinterpret_as_s64(vx_load(&src[i + step / 2]) == zero);
        }

    // N.B the value is incremented by -1 (0xF...F) for each value
    nz = i + (int)v_reduce_sum(sum1 + sum2);
    v_cleanup();
#endif
    return nz + countNonZero_(src + i, len - i);
}
