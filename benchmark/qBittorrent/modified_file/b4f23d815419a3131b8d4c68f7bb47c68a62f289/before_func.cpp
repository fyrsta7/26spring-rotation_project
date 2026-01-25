// Open the parent directory of the given path with a file manager and select
// (if possible) the item at the given path
void Utils::Misc::openFolderSelect(const QString &absolutePath)
{
    const QString path = Utils::Fs::fromNativePath(absolutePath);
    // If the item to select doesn't exist, try to open its parent
    if (!QFileInfo(path).exists()) {
        openPath(path.left(path.lastIndexOf("/")));
        return;
    }
#ifdef Q_OS_WIN
    HRESULT hresult = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    PIDLIST_ABSOLUTE pidl = ::ILCreateFromPathW(reinterpret_cast<PCTSTR>(Utils::Fs::toNativePath(path).utf16()));
    if (pidl) {
        ::SHOpenFolderAndSelectItems(pidl, 0, nullptr, 0);
        ::ILFree(pidl);
    }
    if ((hresult == S_OK) || (hresult == S_FALSE))
        ::CoUninitialize();
#elif defined(Q_OS_UNIX) && !defined(Q_OS_MAC)
    QProcess proc;
    proc.start("xdg-mime", QStringList() << "query" << "default" << "inode/directory");
    proc.waitForFinished();
    QString output = proc.readLine().simplified();
    if ((output == "dolphin.desktop") || (output == "org.kde.dolphin.desktop"))
        proc.startDetached("dolphin", QStringList() << "--select" << Utils::Fs::toNativePath(path));
    else if ((output == "nautilus.desktop") || (output == "org.gnome.Nautilus.desktop")
             || (output == "nautilus-folder-handler.desktop"))
        proc.startDetached("nautilus", QStringList() << "--no-desktop" << Utils::Fs::toNativePath(path));
    else if (output == "nemo.desktop")
        proc.startDetached("nemo", QStringList() << "--no-desktop" << Utils::Fs::toNativePath(path));
    else if ((output == "konqueror.desktop") || (output == "kfmclient_dir.desktop"))
        proc.startDetached("konqueror", QStringList() << "--select" << Utils::Fs::toNativePath(path));
    else
        // "caja" manager can't pinpoint the file, see: https://github.com/qbittorrent/qBittorrent/issues/5003
        openPath(path.left(path.lastIndexOf("/")));
#else
    openPath(path.left(path.lastIndexOf("/")));
#endif
}