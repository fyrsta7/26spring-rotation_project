    constexpr float a = 0.00279491f;
    constexpr float b = 1.15907984f;
    float c = (b / sqrt(1 + a)) - 1;
    return ((b * __builtin_ia32_rsqrtps(x + a)) - c) * x;
}

// Linearize v1 and v2, lerp them by mix factor, then convert back.
// The output is entirely v1 when mix = 0 and entirely v2 when mix = 1
