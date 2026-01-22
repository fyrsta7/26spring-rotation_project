                SetPixel(X, Y, (UCHAR)BackColor);
        }
    }
}

VOID
NTAPI
PreserveRow(
    _In_ ULONG CurrentTop,
    _In_ ULONG TopDelta,
    _In_ BOOLEAN Restore)
{
    PUCHAR OldPosition, NewPosition;
    ULONG PixelCount = TopDelta * SCREEN_WIDTH;

    if (Restore)
    {
        /* Restore the row by copying back the contents saved off-screen */
        OldPosition = (PUCHAR)(FrameBuffer + FB_OFFSET(0, SCREEN_HEIGHT));
        NewPosition = (PUCHAR)(FrameBuffer + FB_OFFSET(0, CurrentTop));
    }
    else
    {
        /* Preserve the row by saving its contents off-screen */
        OldPosition = (PUCHAR)(FrameBuffer + FB_OFFSET(0, CurrentTop));
        NewPosition = (PUCHAR)(FrameBuffer + FB_OFFSET(0, SCREEN_HEIGHT));
