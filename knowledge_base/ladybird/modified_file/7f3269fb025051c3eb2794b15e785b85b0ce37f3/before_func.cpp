            return false;
    }
    return true;
}

bool Utf8View::contains(u32 needle) const
{
    for (u32 code_point : *this) {
