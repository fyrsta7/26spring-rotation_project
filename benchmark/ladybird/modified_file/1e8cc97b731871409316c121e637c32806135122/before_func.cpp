String String::from_utf8_with_replacement_character(StringView view)
{
    StringBuilder builder;

    for (auto c : Utf8View { view })
        builder.append_code_point(c);

    return builder.to_string_without_validation();
}