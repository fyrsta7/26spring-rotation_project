void test_fatfs_rw_speed(const char* filename, void* buf, size_t buf_size, size_t file_size, bool is_write)
{
    const size_t buf_count = file_size / buf_size;

    FILE* f = fopen(filename, (is_write) ? "wb" : "rb");
    TEST_ASSERT_NOT_NULL(f);

    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);
    for (size_t n = 0; n < buf_count; ++n) {
        if (is_write) {
            TEST_ASSERT_EQUAL(buf_size, write(fileno(f), buf, buf_size));
        } else {
            if (read(fileno(f), buf, buf_size) != buf_size) {
                printf("reading at n=%d, eof=%d", n, feof(f));
                TEST_FAIL();
            }
        }
    }

    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);

    TEST_ASSERT_EQUAL(0, fclose(f));

    float t_s = tv_end.tv_sec - tv_start.tv_sec + 1e-6f * (tv_end.tv_usec - tv_start.tv_usec);
    printf("%s %d bytes (block size %d) in %.3fms (%.3f MB/s)\n",
            (is_write)?"Wrote":"Read", file_size, buf_size, t_s * 1e3,
                    file_size / (1024.0f * 1024.0f * t_s));
}
