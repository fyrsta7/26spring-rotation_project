            return false;
    }
    return true;
}

bool Utf8View::contains(u32 needle) const
{
    if (needle <= 0x7f) {
        // OPTIMIZATION: Fast path for ASCII
        for (u8 code_point : as_string()) {
            if (code_point == needle)
                return true;
        }
    } else {
        for (u32 code_point : *this) {
            if (code_point == needle)
                return true;
