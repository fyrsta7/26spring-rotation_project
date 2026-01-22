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
        PLUG_CB_PAUSEDEBUG pauseInfo = { nullptr };
        plugincbcall(CB_PAUSEDEBUG, &pauseInfo);
        dbgsetforeground();
        wait(WAITID_RUN);
    }
    else
    {
        //insert the thread stack as a dummy page to prevent cache misses (issue #1475)
        NT_TIB tib;
        if(ThreadGetTib(ThreadGetLocalBase(dwThreadId), &tib))
        {
            MEMPAGE page;
            auto limit = duint(tib.StackLimit);
            auto base = duint(tib.StackBase);
            sprintf_s(page.info, GuiTranslateText(QT_TRANSLATE_NOOP("DBG", "Thread %X Stack")), dwThreadId);
            page.mbi.BaseAddress = page.mbi.AllocationBase = tib.StackLimit;
            page.mbi.Protect = page.mbi.AllocationProtect = PAGE_READWRITE;
            page.mbi.RegionSize = base - limit;
            page.mbi.State = MEM_COMMIT;
            page.mbi.Type = MEM_PRIVATE;
