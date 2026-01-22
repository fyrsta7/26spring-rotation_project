
    return Color(r.value(), g.value(), b.value(), a.value());
}

Color Color::mixed_with(Color other, float weight) const
{
    if (alpha() == other.alpha() || with_alpha(0) == other.with_alpha(0)) {
        return Gfx::Color {
            round_to<u8>(mix<float>(red(), other.red(), weight)),
            round_to<u8>(mix<float>(green(), other.green(), weight)),
            round_to<u8>(mix<float>(blue(), other.blue(), weight)),
            round_to<u8>(mix<float>(alpha(), other.alpha(), weight)),
        };
    }
    // Fallback to slower, but more visually pleasing premultiplied alpha mix.
    // This is needed for linear-gradient()s in LibWeb.
    auto mixed_alpha = mix<float>(alpha(), other.alpha(), weight);
    auto premultiplied_mix_channel = [&](float channel, float other_channel, float weight) {
        return round_to<u8>(mix<float>(channel * alpha(), other_channel * other.alpha(), weight) / mixed_alpha);
    };
    return Gfx::Color {
        premultiplied_mix_channel(red(), other.red(), weight),
        premultiplied_mix_channel(green(), other.green(), weight),
