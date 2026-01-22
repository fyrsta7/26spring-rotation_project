    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result_pos, size_t input_rows_count) override
    {
        /// Choose JSONParser.
#if USE_SIMDJSON
        if (context.getSettings().allow_simdjson && Cpu::CpuFlagsCache::have_AVX2)
        {
            Executor<SimdJSONParser>::run(block, arguments, result_pos, input_rows_count);
            return;
        }
#endif
#if USE_RAPIDJSON
        Executor<RapidJSONParser>::run(block, arguments, result_pos, input_rows_count);
#else
        Executor<DummyJSONParser>::run(block, arguments, result_pos, input_rows_count);
#endif
    }
