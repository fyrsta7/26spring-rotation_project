    void writeTraceInfo(TimerType timer_type, int /* sig */, siginfo_t * /* info */, void * context)
    {
        constexpr size_t buf_size = sizeof(char) + // TraceCollector stop flag
                                    8 * sizeof(char) + // maximum VarUInt length for string size
                                    QUERY_ID_MAX_LEN * sizeof(char) + // maximum query_id length
                                    sizeof(StackTrace) + // collected stack trace
                                    sizeof(TimerType); // timer type
        char buffer[buf_size];
        WriteBufferFromFileDescriptor out(trace_pipe.fds_rw[1], buf_size, buffer);

        StringRef query_id = CurrentThread::getQueryId();
        query_id.size = std::min(query_id.size, QUERY_ID_MAX_LEN);

        const auto signal_context = *reinterpret_cast<ucontext_t *>(context);
        const StackTrace stack_trace(signal_context);

        writeChar(false, out);
        writeStringBinary(query_id, out);
        writePODBinary(stack_trace, out);
        writePODBinary(timer_type, out);
        out.next();
    }
