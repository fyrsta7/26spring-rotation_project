                Ir->Event.KeyEvent.wVirtualKeyCode == VK_F3)  /* F3 */
            {
                if (ConfirmQuit(Ir))
                    return QUIT_PAGE;
                else
                    return CHECK_FILE_SYSTEM_PAGE;
            }
            else if (Ir->Event.KeyEvent.uChar.AsciiChar == VK_RETURN) /* ENTER */
            {
                return CHECK_FILE_SYSTEM_PAGE;
            }
        }
    }
    else if (!NT_SUCCESS(Status))
    {
        DPRINT1("ChkdskPartition() failed with status 0x%08lx\n", Status);

        RtlStringCbPrintfA(Buffer,
                           sizeof(Buffer),
                           "ChkDsk detected some disk errors.\n(Status 0x%08lx).\n",
                           Status);

        PopupError(Buffer,
                   MUIGetString(STRING_CONTINUE),
                   Ir, POPUP_WAIT_ENTER);
    }

    PartEntry->NeedsCheck = FALSE;
    return CHECK_FILE_SYSTEM_PAGE;
}


static BOOLEAN
IsValidPath(
    IN PCWSTR InstallDir)
{
    UINT i, Length;

    Length = wcslen(InstallDir);

    // TODO: Add check for 8.3 too.

    /* Path must be at least 2 characters long */
//    if (Length < 2)
//        return FALSE;

    /* Path must start with a backslash */
//    if (InstallDir[0] != L'\\')
//        return FALSE;

    /* Path must not end with a backslash */
    if (InstallDir[Length - 1] == L'\\')
        return FALSE;

    /* Path must not contain whitespace characters */
    for (i = 0; i < Length; i++)
    {
        if (iswspace(InstallDir[i]))
            return FALSE;
    }

    /* Path component must not end with a dot */
    for (i = 0; i < Length; i++)
    {
        if (InstallDir[i] == L'\\' && i > 0)
        {
            if (InstallDir[i - 1] == L'.')
                return FALSE;
        }
    }

    if (InstallDir[Length - 1] == L'.')
        return FALSE;

    return TRUE;
}


/*
 * Displays the InstallDirectoryPage.
 *
 * Next pages:
 *  PrepareCopyPage
 *  QuitPage
 *
 * RETURNS
 *   Number of the next page.
 */
static PAGE_NUMBER
InstallDirectoryPage(PINPUT_RECORD Ir)
{
    NTSTATUS Status;
    ULONG Length, Pos;
    WCHAR c;
    WCHAR InstallDir[MAX_PATH];

    /* We do not need the filesystem list anymore */
    ResetFileSystemList();

    if (PartitionList == NULL || InstallPartition == NULL)
    {
        /* FIXME: show an error dialog */
        return QUIT_PAGE;
    }

    // if (IsUnattendedSetup)
    if (RepairUpdateFlag)
        wcscpy(InstallDir, CurrentInstallation->PathComponent); // SystemNtPath
    else if (USetupData.InstallationDirectory[0])
        wcscpy(InstallDir, USetupData.InstallationDirectory);
    else
        wcscpy(InstallDir, L"\\ReactOS");

