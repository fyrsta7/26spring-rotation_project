void _dbg_dbgtraceexecute(duint CIP)
{
    if(TraceRecord.getTraceRecordType(CIP) != TraceRecordManager::TraceRecordType::TraceRecordNone)
    {
        Capstone disassembler;
        unsigned char buffer[16];
        duint size;
        if(MemRead(CIP, buffer, 16))
        {
            TraceRecord.increaseInstructionCounter();
            disassembler.Disassemble(CIP, buffer);
            size = disassembler.Success() ? disassembler.Size() : 1;
            TraceRecord.TraceExecute(CIP, size);
        }
        else
        {
            duint base = MemFindBaseAddr(CIP, &size);
            if(CIP - base + 16 > size) // Corner case where CIP is near the end of the page
            {
                size = base + size - CIP;
                if(MemRead(CIP, buffer, size))
                {
                    TraceRecord.increaseInstructionCounter();
                    disassembler.Disassemble(CIP, buffer, size);
                    size = disassembler.Success() ? disassembler.Size() : 1;
                    TraceRecord.TraceExecute(CIP, size);
                    return;
                }
            }
            // if we reaches here, then the executable had executed an invalid address. Don't trace it.
        }
    }
    else
        TraceRecord.increaseInstructionCounter();
}
