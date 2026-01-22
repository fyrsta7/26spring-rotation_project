ZSTD_decompressSequences_body( ZSTD_DCtx* dctx,
                               void* dst, size_t maxDstSize,
                         const void* seqStart, size_t seqSize, int nbSeq,
                         const ZSTD_longOffset_e isLongOffset,
                         const int frame)
{
    const BYTE* ip = (const BYTE*)seqStart;
    const BYTE* const iend = ip + seqSize;
    BYTE* const ostart = (BYTE*)dst;
    BYTE* const oend = ostart + maxDstSize;
    BYTE* op = ostart;
    const BYTE* litPtr = dctx->litPtr;
    const BYTE* const litEnd = litPtr + dctx->litSize;
    const BYTE* const prefixStart = (const BYTE*) (dctx->prefixStart);
    const BYTE* const vBase = (const BYTE*) (dctx->virtualStart);
    const BYTE* const dictEnd = (const BYTE*) (dctx->dictEnd);
    DEBUGLOG(5, "ZSTD_decompressSequences_body");
    (void)frame;

    /* Regen sequences */
    if (nbSeq) {
        seqState_t seqState;
        size_t error = 0;
        dctx->fseEntropy = 1;
        { U32 i; for (i=0; i<ZSTD_REP_NUM; i++) seqState.prevOffset[i] = dctx->entropy.rep[i]; }
        RETURN_ERROR_IF(
            ERR_isError(BIT_initDStream(&seqState.DStream, ip, iend-ip)),
            corruption_detected, "");
        ZSTD_initFseState(&seqState.stateLL, &seqState.DStream, dctx->LLTptr);
        ZSTD_initFseState(&seqState.stateOffb, &seqState.DStream, dctx->OFTptr);
        ZSTD_initFseState(&seqState.stateML, &seqState.DStream, dctx->MLTptr);
        assert(dst != NULL);

        ZSTD_STATIC_ASSERT(
                BIT_DStream_unfinished < BIT_DStream_completed &&
                BIT_DStream_endOfBuffer < BIT_DStream_completed &&
                BIT_DStream_completed < BIT_DStream_overflow);

#if defined(__GNUC__) && defined(__x86_64__)
        /* Align the decompression loop to 32 + 16 bytes.
         *
         * zstd compiled with gcc-9 on an Intel i9-9900k shows 10% decompression
         * speed swings based on the alignment of the decompression loop. This
         * performance swing is caused by parts of the decompression loop falling
         * out of the DSB. The entire decompression loop should fit in the DSB,
         * when it can't we get much worse performance. You can measure if you've
         * hit the good case or the bad case with this perf command for some
         * compressed file test.zst:
         *
         *   perf stat -e cycles -e instructions -e idq.all_dsb_cycles_any_uops \
         *             -e idq.all_mite_cycles_any_uops -- ./zstd -tq test.zst
         *
         * If you see most cycles served out of the MITE you've hit the bad case.
         * If you see most cycles served out of the DSB you've hit the good case.
         * If it is pretty even then you may be in an okay case.
         *
         * I've been able to reproduce this issue on the following CPUs:
         *   - Kabylake: Macbook Pro (15-inch, 2019) 2.4 GHz Intel Core i9
         *               Use Instruments->Counters to get DSB/MITE cycles.
         *               I never got performance swings, but I was able to
         *               go from the good case of mostly DSB to half of the
         *               cycles served from MITE.
         *   - Coffeelake: Intel i9-9900k
         *
         * I haven't been able to reproduce the instability or DSB misses on any
         * of the following CPUS:
         *   - Haswell
         *   - Broadwell: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GH
         *   - Skylake
         *
         * If you are seeing performance stability this script can help test.
         * It tests on 4 commits in zstd where I saw performance change.
         *
         *   https://gist.github.com/terrelln/9889fc06a423fd5ca6e99351564473f4
         */
        __asm__(".p2align 5");
        __asm__("nop");
        __asm__(".p2align 4");
#endif
        for ( ; ; ) {
            seq_t const sequence = ZSTD_decodeSequence(&seqState, isLongOffset, ZSTD_p_noPrefetch);
            size_t const oneSeqSize = ZSTD_execSequence(op, oend, sequence, &litPtr, litEnd, prefixStart, vBase, dictEnd);
#if defined(FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION) && defined(FUZZING_ASSERT_VALID_SEQUENCE)
            assert(!ZSTD_isError(oneSeqSize));
            if (frame) ZSTD_assertValidSequence(dctx, op, oend, sequence, prefixStart, vBase);
#endif
            DEBUGLOG(6, "regenerated sequence size : %u", (U32)oneSeqSize);
            BIT_reloadDStream(&(seqState.DStream));
            op += oneSeqSize;
            /* gcc and clang both don't like early returns in this loop.
             * Instead break and check for an error at the end of the loop.
             */
