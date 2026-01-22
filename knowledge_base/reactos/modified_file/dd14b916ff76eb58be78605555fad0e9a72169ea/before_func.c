BOOLEAN
NTAPI
InbvDriverInitialize(IN PLOADER_PARAMETER_BLOCK LoaderBlock,
                     IN ULONG Count)
{
    PCHAR CommandLine;
    BOOLEAN CustomLogo = FALSE;
    ULONG i;
    extern BOOLEAN ExpInTextModeSetup;

    /* Quit if we're already installed */
    if (InbvBootDriverInstalled) return TRUE;

    /* Initialize the lock and check the current display state */
    KeInitializeSpinLock(&BootDriverLock);
    if (InbvDisplayState == INBV_DISPLAY_STATE_OWNED)
    {
        /* Check if we have a custom boot logo */
        CommandLine = _strupr(LoaderBlock->LoadOptions);
        CustomLogo = strstr(CommandLine, "BOOTLOGO") ? TRUE: FALSE;
    }

    /* For SetupLDR, don't reset the BIOS Display -- FIXME! */
    if (ExpInTextModeSetup) CustomLogo = TRUE;

    /* Initialize the video */
    InbvBootDriverInstalled = VidInitialize(!CustomLogo);
    if (InbvBootDriverInstalled)
    {
        /* Find bitmap resources in the kernel */
        ResourceCount = Count;
        for (i = 0; i < Count; i++)
        {
            /* Do the lookup */
            ResourceList[i] = FindBitmapResource(LoaderBlock, i);
        }

        /* Set the progress bar ranges */
        InbvSetProgressBarSubset(0, 100);
    }

    /* Return install state */
    return InbvBootDriverInstalled;
}
