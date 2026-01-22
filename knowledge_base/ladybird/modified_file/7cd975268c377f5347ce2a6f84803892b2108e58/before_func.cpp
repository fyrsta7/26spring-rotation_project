    }
    return {};
}

// Partial implementation of the font-matching algorithm: https://www.w3.org/TR/css-fonts-4/#font-matching-algorithm
// FIXME: This should be replaced by the full CSS font selection algorithm.
RefPtr<Gfx::Font const> StyleComputer::font_matching_algorithm(FontFaceKey const& key, float font_size_in_pt) const
{
    // If a font family match occurs, the user agent assembles the set of font faces in that family and then
    // narrows the set to a single face using other font properties in the order given below.
    Vector<MatchingFontCandidate> matching_family_fonts;
    for (auto const& font_key_and_loader : m_loaded_fonts) {
        if (font_key_and_loader.key.family_name.equals_ignoring_ascii_case(key.family_name))
            matching_family_fonts.empend(font_key_and_loader.key, font_key_and_loader.value.ptr());
    }
    Gfx::FontDatabase::the().for_each_typeface([&](Gfx::Typeface const& typeface) {
        if (typeface.family().equals_ignoring_ascii_case(key.family_name)) {
            matching_family_fonts.empend(
                FontFaceKey {
                    .family_name = MUST(FlyString::from_deprecated_fly_string(typeface.family())),
                    .weight = static_cast<int>(typeface.weight()),
                    .slope = typeface.slope(),
                },
                &typeface);
        }
    });
    quick_sort(matching_family_fonts, [](auto const& a, auto const& b) {
        return a.key.weight < b.key.weight;
    });
    // FIXME: 1. font-stretch is tried first.
    // FIXME: 2. font-style is tried next.
    // We don't have complete support of italic and oblique fonts, so matching on font-style can be simplified to:
    // If a matching slope is found, all faces which don't have that matching slope are excluded from the matching set.
    auto style_it = find_if(matching_family_fonts.begin(), matching_family_fonts.end(),
        [&](auto const& matching_font_candidate) { return matching_font_candidate.key.slope == key.slope; });
    if (style_it != matching_family_fonts.end()) {
        matching_family_fonts.remove_all_matching([&](auto const& matching_font_candidate) {
            return matching_font_candidate.key.slope != key.slope;
        });
    }
    // 3. font-weight is matched next.
    // If the desired weight is inclusively between 400 and 500, weights greater than or equal to the target weight
    // are checked in ascending order until 500 is hit and checked, followed by weights less than the target weight
    // in descending order, followed by weights greater than 500, until a match is found.
    if (key.weight >= 400 && key.weight <= 500) {
        auto it = find_if(matching_family_fonts.begin(), matching_family_fonts.end(),
            [&](auto const& matching_font_candidate) { return matching_font_candidate.key.weight >= key.weight; });
        for (; it != matching_family_fonts.end() && it->key.weight <= 500; ++it) {
            if (auto found_font = it->font_with_point_size(font_size_in_pt))
                return found_font;
        }
        if (auto found_font = find_matching_font_weight_descending(matching_family_fonts, key.weight, font_size_in_pt, false))
            return found_font;
        for (; it != matching_family_fonts.end(); ++it) {
            if (auto found_font = it->font_with_point_size(font_size_in_pt))
                return found_font;
        }
    }
    // If the desired weight is less than 400, weights less than or equal to the desired weight are checked in descending order
    // followed by weights above the desired weight in ascending order until a match is found.
    if (key.weight < 400) {
        if (auto found_font = find_matching_font_weight_descending(matching_family_fonts, key.weight, font_size_in_pt, true))
            return found_font;
        if (auto found_font = find_matching_font_weight_ascending(matching_family_fonts, key.weight, font_size_in_pt, false))
            return found_font;
    }
    // If the desired weight is greater than 500, weights greater than or equal to the desired weight are checked in ascending order
    // followed by weights below the desired weight in descending order until a match is found.
    if (key.weight > 500) {
        if (auto found_font = find_matching_font_weight_ascending(matching_family_fonts, key.weight, font_size_in_pt, true))
