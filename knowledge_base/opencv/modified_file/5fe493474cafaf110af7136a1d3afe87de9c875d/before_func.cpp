TEST(GaussianBlur)
{
    for (int size = 1000; size <= 4000; size += 1000)
    {
        SUBTEST << size << 'x' << size << ", 8UC4";

        Mat src, dst;
        
        gen(src, size, size, CV_8UC4, 0, 256);

        GaussianBlur(src, dst, Size(3, 3), 1);

        CPU_ON;
        GaussianBlur(src, dst, Size(3, 3), 1);
        CPU_OFF;

        gpu::GpuMat d_src(src);
        gpu::GpuMat d_dst(src.size(), src.type());
        gpu::GpuMat d_buf;

        gpu::GaussianBlur(d_src, d_dst, Size(3, 3), d_buf, 1);

        GPU_ON;
        gpu::GaussianBlur(d_src, d_dst, Size(3, 3), d_buf, 1);
        GPU_OFF;
    }
}
