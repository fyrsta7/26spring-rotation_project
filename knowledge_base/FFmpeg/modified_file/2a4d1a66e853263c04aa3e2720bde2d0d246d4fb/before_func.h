    if (!(v & 0x3)) {
        v >>= 2;
        c += 2;
    }
    c -= v & 0x1;

    return c;
}
#else
static av_always_inline av_const int ff_ctz_c( int v )
