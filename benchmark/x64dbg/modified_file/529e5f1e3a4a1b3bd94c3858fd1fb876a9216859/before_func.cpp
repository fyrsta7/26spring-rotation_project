void ThreadExit(DWORD ThreadId)
{
    EXCLUSIVE_ACQUIRE(LockThreads);

    // Don't use a foreach loop here because of the iterator erase() call
    for(auto itr = threadList.begin(); itr != threadList.end(); itr++)
    {
        if(itr->first == ThreadId)
        {
            threadList.erase(itr);
            break;
        }
    }

    EXCLUSIVE_RELEASE();
    GuiUpdateThreadView();
}