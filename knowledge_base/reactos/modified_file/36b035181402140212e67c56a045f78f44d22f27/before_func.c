                                wParam,
                                lParam,
                                5000,
                                TRUE,
                                MSQ_ISHOOK,
                                &uResult);

    return NT_SUCCESS(Status) ? uResult : 0;
}

/*
  Called from inside kernel space.
 */
LRESULT
FASTCALL
co_HOOK_CallHooks(INT HookId, INT Code, WPARAM wParam, LPARAM lParam)
{
    PHOOK Hook, SaveHook;
    PTHREADINFO pti;
    PCLIENTINFO ClientInfo;
    PHOOKTABLE Table;
    LRESULT Result;
    PWINSTATION_OBJECT WinStaObj;
    NTSTATUS Status;

    ASSERT(WH_MINHOOK <= HookId && HookId <= WH_MAXHOOK);

    /* FIXME! Check pDeskInfo->fsHooks for global hooks! */
    if (!ISITHOOKED(HookId))
    {
        return 0;
    }

    pti = PsGetCurrentThreadWin32Thread();
    if (!pti)
    {
        Table = NULL;
    }
    else
    {
        Table = MsqGetHooks(pti->MessageQueue);
    }

    if (NULL == Table || ! (Hook = IntGetFirstValidHook(Table, HookId)))
    {
        /* try global table */
        Table = GlobalHooks;
        if (NULL == Table || ! (Hook = IntGetFirstValidHook(Table, HookId)))
        {
            return 0;  /* no hook set */
        }
    }

    if ((Hook->Thread != PsGetCurrentThread()) && (Hook->Thread != NULL))
    {
        DPRINT1("\nHook found by Id and posted to Thread! %d\n",HookId );
        /* Post it in message queue. */
        return IntCallLowLevelHook(Hook, Code, wParam, lParam);
    }

    Table->Counts[HOOKID_TO_INDEX(HookId)]++;
    if (Table != GlobalHooks && GlobalHooks != NULL)
    {
        GlobalHooks->Counts[HOOKID_TO_INDEX(HookId)]++;
    }

    ClientInfo = GetWin32ClientInfo();
    SaveHook = ClientInfo->phkCurrent;
    ClientInfo->phkCurrent = Hook;     /* Load the call. */

    Result = co_IntCallHookProc(HookId,
                                Code,
                                wParam,
                                lParam,
                                Hook->Proc,
                                Hook->Ansi,
                                &Hook->ModuleName);

    ClientInfo->phkCurrent = SaveHook;

    Status = IntValidateWindowStationHandle(PsGetCurrentProcess()->Win32WindowStation,
                                            KernelMode,
                                            0,
                                            &WinStaObj);

