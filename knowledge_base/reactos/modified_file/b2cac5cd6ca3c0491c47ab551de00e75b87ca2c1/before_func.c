    va_start(ap, fmt);
    vsprintf(buffer, fmt, ap);
    va_end(ap);

    coPos.X = x;
    coPos.Y = y;

    Length = (SHORT)strlen(buffer);
    if (Length > len - 1)
        Length = len - 1;

    WriteConsoleOutputCharacterA(StdOutput,
                                 buffer,
                                 Length,
                                 coPos,
                                 &Written);

    coPos.X += Length;

    if (len > Length)
    {
        FillConsoleOutputCharacterA(StdOutput,
                                    ' ',
                                    len - Length,
                                    coPos,
                                    &Written);
    }
}

VOID
CONSOLE_SetStyledText(
    IN SHORT x,
    IN SHORT y,
    IN INT Flags,
    IN LPCSTR Text)
{
    COORD coPos;
    DWORD Length;

    coPos.X = x;
    coPos.Y = y;

    Length = (ULONG)strlen(Text);

    if (Flags & TEXT_TYPE_STATUS)
    {
        coPos.X = x;
        coPos.Y = yScreen - 1;
    }
    else /* TEXT_TYPE_REGULAR (Default) */
    {
        coPos.X = x;
        coPos.Y = y;
    }

    if (Flags & TEXT_ALIGN_CENTER)
    {
        coPos.X = (xScreen - Length) / 2;
    }
    else if(Flags & TEXT_ALIGN_RIGHT)
    {
        coPos.X = coPos.X - Length;

        if (Flags & TEXT_PADDING_SMALL)
        {
            coPos.X -= 1;
        }
        else if (Flags & TEXT_PADDING_MEDIUM)
        {
            coPos.X -= 2;
        }
        else if (Flags & TEXT_PADDING_BIG)
        {
            coPos.X -= 3;
        }
    }
    else /* TEXT_ALIGN_LEFT (Default) */
    {
        if (Flags & TEXT_PADDING_SMALL)
        {
            coPos.X += 1;
        }
        else if (Flags & TEXT_PADDING_MEDIUM)
