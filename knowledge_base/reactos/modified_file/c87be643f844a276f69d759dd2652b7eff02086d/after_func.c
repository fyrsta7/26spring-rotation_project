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
    PULONG OldPosition, NewPosition;
    ULONG PixelCount = TopDelta * (SCREEN_WIDTH / sizeof(ULONG));

    if (Restore)
    {
        /* Restore the row by copying back the contents saved off-screen */
        OldPosition = (PULONG)(FrameBuffer + FB_OFFSET(0, SCREEN_HEIGHT));
        NewPosition = (PULONG)(FrameBuffer + FB_OFFSET(0, CurrentTop));
    }
    else
    {
        /* Preserve the row by saving its contents off-screen */
        OldPosition = (PULONG)(FrameBuffer + FB_OFFSET(0, CurrentTop));
        NewPosition = (PULONG)(FrameBuffer + FB_OFFSET(0, SCREEN_HEIGHT));
