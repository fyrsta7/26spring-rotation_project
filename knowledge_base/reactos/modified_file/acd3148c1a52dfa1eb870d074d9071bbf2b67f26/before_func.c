VOID
CONSOLE_SetUnderlinedTextXY(
    IN SHORT x,
    IN SHORT y,
    IN LPCSTR Text)
{
    COORD coPos;
    DWORD Length;
    DWORD Written;

    coPos.X = x;
    coPos.Y = y;

    Length = (ULONG)strlen(Text);

    WriteConsoleOutputCharacterA(StdOutput,
                                 Text,
                                 Length,
                                 coPos,
                                 &Written);

    coPos.Y++;
    FillConsoleOutputCharacterA(StdOutput,
                                CharDoubleHorizontalLine,
                                Length,
                                coPos,
                                &Written);
}

VOID
CONSOLE_SetStatusTextXV(
    IN SHORT x,
    IN LPCSTR fmt,
    IN va_list args)
{
