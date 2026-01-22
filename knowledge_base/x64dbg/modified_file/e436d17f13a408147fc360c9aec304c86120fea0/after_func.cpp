void _dbg_dbgtraceexecute(duint CIP)
{
    if(TraceRecord.getTraceRecordType(CIP) != TraceRecordManager::TraceRecordType::TraceRecordNone)
    {
        unsigned char buffer[16];
        duint size;
        if(MemRead(CIP, buffer, 16))
        {
            BASIC_INSTRUCTION_INFO basicInfo;
            TraceRecord.increaseInstructionCounter();
            DbgDisasmFastAt(CIP, &basicInfo);
            TraceRecord.TraceExecute(CIP, basicInfo.size);
        }
        else
        {
            duint base = MemFindBaseAddr(CIP, &size);
            if(CIP - base + 16 > size) // Corner case where CIP is near the end of the page
            {
                size = base + size - CIP;
                if(MemRead(CIP, buffer, size))
                {
                    BASIC_INSTRUCTION_INFO basicInfo;
                    TraceRecord.increaseInstructionCounter();
                    DbgDisasmFastAt(CIP, &basicInfo);
                    TraceRecord.TraceExecute(CIP, basicInfo.size);
                    return;
                }
            }
            // if we reaches here, then the executable had executed an invalid address. Don't trace it.
        }
    }
    else
        TraceRecord.increaseInstructionCounter();
}
