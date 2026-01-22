typedef union {
    float f;
    struct {
        uint64_t m : 23;
        uint64_t e : 8;
        uint64_t s : 1;
    };
} float_s_t;

#ifndef NDEBUG
