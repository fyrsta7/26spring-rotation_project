inline int OutputDebugStringF(const char* pszFormat, ...)
{
    int ret = 0;
    if (fPrintToConsole)
    {
        // print to console
        va_list arg_ptr;
        va_start(arg_ptr, pszFormat);
        ret = vprintf(pszFormat, arg_ptr);
        va_end(arg_ptr);
    }
    else
    {
        // print to debug.log
        static FILE* fileout = NULL;
        static int64 nOpenTime = 0;

        if (GetTime()-nOpenTime > 10 * 60)
        {
            if (fileout)
                fclose(fileout);
            char pszFile[MAX_PATH+100];
            GetDataDir(pszFile);
            strlcat(pszFile, "/debug.log", sizeof(pszFile));
            fileout = fopen(pszFile, "a");
            nOpenTime = GetTime();
        }
        if (fileout)
        {
            //// Debug print useful for profiling
            //fprintf(fileout, " %"PRI64d" ", GetTimeMillis());
            va_list arg_ptr;
            va_start(arg_ptr, pszFormat);
            ret = vfprintf(fileout, pszFormat, arg_ptr);
            va_end(arg_ptr);
            fflush(fileout);
        }
    }

#ifdef __WXMSW__
    if (fPrintToDebugger)
    {
        // accumulate a line at a time
        static CCriticalSection cs_OutputDebugStringF;
        CRITICAL_BLOCK(cs_OutputDebugStringF)
        {
            static char pszBuffer[50000];
            static char* pend;
            if (pend == NULL)
                pend = pszBuffer;
            va_list arg_ptr;
            va_start(arg_ptr, pszFormat);
            int limit = END(pszBuffer) - pend - 2;
            int ret = _vsnprintf(pend, limit, pszFormat, arg_ptr);
            va_end(arg_ptr);
            if (ret < 0 || ret >= limit)
            {
                pend = END(pszBuffer) - 2;
                *pend++ = '\n';
            }
            else
                pend += ret;
            *pend = '\0';
            char* p1 = pszBuffer;
            char* p2;
            while (p2 = strchr(p1, '\n'))
            {
                p2++;
                char c = *p2;
                *p2 = '\0';
                OutputDebugStringA(p1);
                *p2 = c;
                p1 = p2;
            }
            if (p1 != pszBuffer)
                memmove(pszBuffer, p1, pend - p1 + 1);
            pend -= (p1 - pszBuffer);
        }
    }
#endif
    return ret;
}
