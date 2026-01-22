}

/// Galois Field multiplication using <x^127 + x^7 + x^2 + x + 1>.
/// Note that x, y, and z are strictly BE.
void galois_multiply(u32 (&_z)[4], u32 const (&_x)[4], u32 const (&_y)[4])
{
    // Note: Copied upfront to stack to avoid memory access in the loop.
    u32 x[4] { _x[0], _x[1], _x[2], _x[3] };
    u32 const y[4] { _y[0], _y[1], _y[2], _y[3] };
    u32 z[4] { 0, 0, 0, 0 };

    // Unrolled by 32, the access in y[3-(i/32)] can be cached throughout the loop.
#pragma GCC unroll 32
    for (ssize_t i = 127; i > -1; --i) {
        auto r = -((y[3 - (i / 32)] >> (i % 32)) & 1);
        z[0] ^= x[0] & r;
        z[1] ^= x[1] & r;
        z[2] ^= x[2] & r;
        z[3] ^= x[3] & r;
        auto a0 = x[0] & 1;
        x[0] >>= 1;
        auto a1 = x[1] & 1;
        x[1] >>= 1;
        x[1] |= a0 << 31;
        auto a2 = x[2] & 1;
        x[2] >>= 1;
        x[2] |= a1 << 31;
        auto a3 = x[3] & 1;
        x[3] >>= 1;
        x[3] |= a2 << 31;

        x[0] ^= 0xe1000000 & -a3;
