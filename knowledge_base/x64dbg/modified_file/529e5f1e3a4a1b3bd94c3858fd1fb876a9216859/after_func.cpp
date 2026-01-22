    // Notify GUI
    GuiUpdateThreadView();
}

void ThreadExit(DWORD ThreadId)
{
    EXCLUSIVE_ACQUIRE(LockThreads);

    // Erase element using native functions
    auto itr = threadList.find(ThreadId);

    if(itr != threadList.end())
        threadList.erase(itr);
