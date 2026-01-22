    dlOnLowResource,
    dlOnProgress,
    dlOnStopBinding,
    dlGetBindInfo,
    dlOnDataAvailable,
    dlOnObjectAvailable
};

static IBindStatusCallback*
CreateDl(HWND Dlg, BOOL *pbCancelled)
{
    IBindStatusCallbackImpl *This;

    This = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(IBindStatusCallbackImpl));
    if (!This) return NULL;

    This->vtbl = &dlVtbl;
    This->ref = 1;
    This->hDialog = Dlg;
    This->pbCancelled = pbCancelled;

    return (IBindStatusCallback*) This;
}

static
DWORD WINAPI
ThreadFunc(LPVOID Context)
{
    IBindStatusCallback *dl;
    WCHAR path[MAX_PATH];
    LPWSTR p;
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    HWND Dlg = (HWND) Context;
    DWORD r, len;
    BOOL bCancelled = FALSE;
    BOOL bTempfile = FALSE;
    BOOL bCab = FALSE;

    /* built the path for the download */
    p = wcsrchr(AppInfo->szUrlDownload, L'/');
    if (!p) goto end;

    len = wcslen(AppInfo->szUrlDownload);
    if (len > 4)
    {
        if (AppInfo->szUrlDownload[len - 4] == '.' &&
            AppInfo->szUrlDownload[len - 3] == 'c' &&
            AppInfo->szUrlDownload[len - 2] == 'a' &&
            AppInfo->szUrlDownload[len - 1] == 'b')
        {
            bCab = TRUE;
            if (!GetCurrentDirectoryW(MAX_PATH, path))
                goto end;
        }
        else
        {
            wcscpy(path, SettingsInfo.szDownloadDir);
        }
    }
    else goto end;

    if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES)
    {
        if (!CreateDirectoryW(path, NULL))
            goto end;
    }

    wcscat(path, L"\\");
    wcscat(path, p + 1);

    /* download it */
    bTempfile = TRUE;
    dl = CreateDl(Context, &bCancelled);
    r = URLDownloadToFileW(NULL, AppInfo->szUrlDownload, path, 0, dl);
    if (dl) IBindStatusCallback_Release(dl);
    if (S_OK != r) goto end;
    else if (bCancelled) goto end;
