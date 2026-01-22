#include <simdutf.h>

namespace AK {

String String::from_utf8_with_replacement_character(StringView view)
{
    if (Utf8View(view).validate())
        return String::from_utf8_without_validation(view.bytes());

    StringBuilder builder;

    for (auto c : Utf8View { view })
