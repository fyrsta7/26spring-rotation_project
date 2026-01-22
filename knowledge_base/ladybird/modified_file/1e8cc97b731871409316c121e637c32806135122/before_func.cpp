#include <simdutf.h>

namespace AK {

String String::from_utf8_with_replacement_character(StringView view)
{
    StringBuilder builder;

    for (auto c : Utf8View { view })
