            *va_arg(args, WCHAR **) = ExtractUntil(str, *(f + 1), &end);
        else if ('S' == *f)
            va_arg(args, ScopedMem<WCHAR> *)->Set(ExtractUntil(str, *(f + 1), &end));
        else if ('$' == *f && !*str)
            continue; // don't fail, if we're indeed at the end of the string
        else if ('%' == *f && *f == *str)
            end = str + 1;
        else if (' ' == *f && str::IsWs(*str))
            end = str + 1;
        else if ('_' == *f) {
