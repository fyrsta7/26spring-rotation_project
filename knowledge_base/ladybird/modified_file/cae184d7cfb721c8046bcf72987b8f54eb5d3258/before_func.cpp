        haystack.characters_without_null_termination() + start, haystack.length() - start,
        needle.characters_without_null_termination(), needle.length());
    return index.has_value() ? (*index + start) : index;
}

Optional<size_t> find_last(StringView haystack, char needle)
{
    for (size_t i = haystack.length(); i > 0; --i) {
