
static bool IsUsingInstallation(DWORD procId) {
    ScopedHandle snap(CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, procId));
    if (snap == INVALID_HANDLE_VALUE)
        return false;

    AutoFreeW libmupdf(path::Join(gInstUninstGlobals.installDir, L"libmupdf.dll"));
    AutoFreeW browserPlugin(GetBrowserPluginPath());

    MODULEENTRY32 mod = {0};
    mod.dwSize = sizeof(mod);
    BOOL cont = Module32First(snap, &mod);
    while (cont) {
        if (path::IsSame(libmupdf, mod.szExePath) || path::IsSame(browserPlugin, mod.szExePath)) {
            return true;
        }
        cont = Module32Next(snap, &mod);
    }

    return false;
