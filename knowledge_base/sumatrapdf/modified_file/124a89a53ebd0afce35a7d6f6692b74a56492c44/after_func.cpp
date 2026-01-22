
static bool IsUsingInstallation(DWORD procId) {
    ScopedHandle snap(CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, procId));
    if (snap == INVALID_HANDLE_VALUE) {
        return false;
    }

    AutoFreeW libmupdf(path::Join(gInstUninstGlobals.installDir, L"libmupdf.dll"));
    AutoFreeW browserPlugin(GetBrowserPluginPath());
    const WCHAR* libmupdfName = path::GetBaseNameNoFree(libmupdf);
    const WCHAR* browserPluginName = path::GetBaseNameNoFree(browserPlugin);

    MODULEENTRY32 mod = {0};
    mod.dwSize = sizeof(mod);
    BOOL cont = Module32First(snap, &mod);
    while (cont) {
        WCHAR* exePath = mod.szExePath;
        const WCHAR* exeName = path::GetBaseNameNoFree(exePath);
        // path::IsSame() is slow so speed up comparison by checking if names are equal first
        if (str::EqI(exeName, libmupdfName) && path::IsSame(libmupdf, exePath)) {
            return true;
        }

        if (str::EqI(exeName, browserPluginName) && path::IsSame(browserPlugin, exePath)) {
            return true;
        }
        cont = Module32Next(snap, &mod);
    }

    return false;
