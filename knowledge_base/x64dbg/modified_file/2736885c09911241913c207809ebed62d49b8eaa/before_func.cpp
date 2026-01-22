    //history
    dbgcleartracestate();
    dbgClearRtuBreakpoints();
    HistoryClear();
}

static void cbCreateThread(CREATE_THREAD_DEBUG_INFO* CreateThread)
{
    ThreadCreate(CreateThread); //update thread list
    DWORD dwThreadId = ((DEBUG_EVENT*)GetDebugData())->dwThreadId;
    hActiveThread = ThreadGetHandle(dwThreadId);

    auto entry = duint(CreateThread->lpStartAddress);
    if(settingboolget("Events", "ThreadEntry"))
    {
        String command;
        command = StringUtils::sprintf("bp %p,\"%s %X\",ss", entry, GuiTranslateText(QT_TRANSLATE_NOOP("DBG", "Thread")), dwThreadId);
        cmddirectexec(command.c_str());
    }

    PLUG_CB_CREATETHREAD callbackInfo;
    callbackInfo.CreateThread = CreateThread;
    callbackInfo.dwThreadId = dwThreadId;
    plugincbcall(CB_CREATETHREAD, &callbackInfo);

    auto symbolic = SymGetSymbolicName(entry);
    if(!symbolic.length())
        symbolic = StringUtils::sprintf("%p", entry);
    dprintf(QT_TRANSLATE_NOOP("DBG", "Thread %X created, Entry: %s\n"), dwThreadId, symbolic.c_str());

    if(settingboolget("Events", "ThreadStart"))
    {
        HistoryClear();
        //update memory map
        MemUpdateMap();
        //update GUI
        DebugUpdateGuiSetStateAsync(GetContextDataEx(hActiveThread, UE_CIP), true);
        //lock
        lock(WAITID_RUN);
        // Plugin callback
