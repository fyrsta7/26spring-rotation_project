    free(cNoiseBuffer[0]);
    free(cNoiseBuffer[1]);
    free(cNoiseBuffer[2]);
    free(cNoiseBuffer[3]);
    free(cNoiseBuffer[4]);
    free(copyBuffer);
    free(cBuffer);
    free(dstBuffer);
    return result;

_output_error:
    result = 1;
    goto _cleanup;
}

/** If useOpaqueAPI, sets param in cctxParams.
 *  Otherwise, sets the param in zc. */
static size_t setCCtxParameter(ZSTD_CCtx* zc, ZSTD_CCtx_params* cctxParams,
                               ZSTD_cParameter param, unsigned value,
                               U32 useOpaqueAPI)
{
    if (useOpaqueAPI) {
        return ZSTD_CCtxParam_setParameter(cctxParams, param, value);
    } else {
        return ZSTD_CCtx_setParameter(zc, param, value);
    }
}

/* Tests for ZSTD_compress_generic() API */
static int fuzzerTests_newAPI(U32 seed, U32 nbTests, unsigned startTest, double compressibility, int bigTests, U32 const useOpaqueAPI)
{
    U32 const maxSrcLog = bigTests ? 24 : 22;
    static const U32 maxSampleLog = 19;
    size_t const srcBufferSize = (size_t)1<<maxSrcLog;
    BYTE* cNoiseBuffer[5];
    size_t const copyBufferSize= srcBufferSize + (1<<maxSampleLog);
    BYTE*  const copyBuffer = (BYTE*)malloc (copyBufferSize);
    size_t const cBufferSize   = ZSTD_compressBound(srcBufferSize);
    BYTE*  const cBuffer = (BYTE*)malloc (cBufferSize);
    size_t const dstBufferSize = srcBufferSize;
    BYTE*  const dstBuffer = (BYTE*)malloc (dstBufferSize);
    U32 result = 0;
    U32 testNb = 0;
    U32 coreSeed = seed;
    ZSTD_CCtx* zc = ZSTD_createCCtx();   /* will be reset sometimes */
    ZSTD_DStream* zd = ZSTD_createDStream();   /* will be reset sometimes */
    ZSTD_DStream* const zd_noise = ZSTD_createDStream();
    clock_t const startClock = clock();
    const BYTE* dict = NULL;   /* can keep same dict on 2 consecutive tests */
    size_t dictSize = 0;
    U32 oldTestLog = 0;
    U32 const cLevelMax = bigTests ? (U32)ZSTD_maxCLevel() : g_cLevelMax_smallTests;
    U32 const nbThreadsMax = bigTests ? 5 : 1;
    ZSTD_CCtx_params* cctxParams = ZSTD_createCCtxParams();

    /* allocations */
    cNoiseBuffer[0] = (BYTE*)malloc (srcBufferSize);
    cNoiseBuffer[1] = (BYTE*)malloc (srcBufferSize);
    cNoiseBuffer[2] = (BYTE*)malloc (srcBufferSize);
    cNoiseBuffer[3] = (BYTE*)malloc (srcBufferSize);
    cNoiseBuffer[4] = (BYTE*)malloc (srcBufferSize);
    CHECK (!cNoiseBuffer[0] || !cNoiseBuffer[1] || !cNoiseBuffer[2] || !cNoiseBuffer[3] || !cNoiseBuffer[4] ||
           !copyBuffer || !dstBuffer || !cBuffer || !zc || !zd || !zd_noise ,
           "Not enough memory, fuzzer tests cancelled");

    /* Create initial samples */
    RDG_genBuffer(cNoiseBuffer[0], srcBufferSize, 0.00, 0., coreSeed);    /* pure noise */
    RDG_genBuffer(cNoiseBuffer[1], srcBufferSize, 0.05, 0., coreSeed);    /* barely compressible */
    RDG_genBuffer(cNoiseBuffer[2], srcBufferSize, compressibility, 0., coreSeed);
    RDG_genBuffer(cNoiseBuffer[3], srcBufferSize, 0.95, 0., coreSeed);    /* highly compressible */
    RDG_genBuffer(cNoiseBuffer[4], srcBufferSize, 1.00, 0., coreSeed);    /* sparse content */
    memset(copyBuffer, 0x65, copyBufferSize);                             /* make copyBuffer considered initialized */
    CHECK_Z( ZSTD_initDStream_usingDict(zd, NULL, 0) );   /* ensure at least one init */

    /* catch up testNb */
    for (testNb=1; testNb < startTest; testNb++)
        FUZ_rand(&coreSeed);

    /* test loop */
    for ( ; (testNb <= nbTests) || (FUZ_GetClockSpan(startClock) < g_clockTime) ; testNb++ ) {
        U32 lseed;
        const BYTE* srcBuffer;
        size_t totalTestSize, totalGenSize, cSize;
        XXH64_state_t xxhState;
        U64 crcOrig;
        U32 resetAllowed = 1;
        size_t maxTestSize;

        /* init */
        if (nbTests >= testNb) { DISPLAYUPDATE(2, "\r%6u/%6u    ", testNb, nbTests); }
        else { DISPLAYUPDATE(2, "\r%6u          ", testNb); }
        FUZ_rand(&coreSeed);
        lseed = coreSeed ^ prime32;

        /* states full reset (deliberately not synchronized) */
        /* some issues can only happen when reusing states */
        if ((FUZ_rand(&lseed) & 0xFF) == 131) {
            DISPLAYLEVEL(5, "Creating new context \n");
            ZSTD_freeCCtx(zc);
            zc = ZSTD_createCCtx();
            CHECK(zc==NULL, "ZSTD_createCCtx allocation error");
            resetAllowed=0;
        }
        if ((FUZ_rand(&lseed) & 0xFF) == 132) {
            ZSTD_freeDStream(zd);
            zd = ZSTD_createDStream();
            CHECK(zd==NULL, "ZSTD_createDStream allocation error");
            ZSTD_initDStream_usingDict(zd, NULL, 0);  /* ensure at least one init */
        }

        /* srcBuffer selection [0-4] */
        {   U32 buffNb = FUZ_rand(&lseed) & 0x7F;
            if (buffNb & 7) buffNb=2;   /* most common : compressible (P) */
            else {
                buffNb >>= 3;
                if (buffNb & 7) {
                    const U32 tnb[2] = { 1, 3 };   /* barely/highly compressible */
                    buffNb = tnb[buffNb >> 3];
                } else {
                    const U32 tnb[2] = { 0, 4 };   /* not compressible / sparse */
                    buffNb = tnb[buffNb >> 3];
            }   }
            srcBuffer = cNoiseBuffer[buffNb];
        }

        /* compression init */
        CHECK_Z( ZSTD_CCtx_loadDictionary(zc, NULL, 0) );   /* cancel previous dict /*/
        if ((FUZ_rand(&lseed)&1) /* at beginning, to keep same nb of rand */
            && oldTestLog /* at least one test happened */ && resetAllowed) {
            maxTestSize = FUZ_randomLength(&lseed, oldTestLog+2);
            if (maxTestSize >= srcBufferSize) maxTestSize = srcBufferSize-1;
            {   int const compressionLevel = (FUZ_rand(&lseed) % 5) + 1;
                CHECK_Z (setCCtxParameter(zc, cctxParams, ZSTD_p_compressionLevel, compressionLevel, useOpaqueAPI) );
            }
        } else {
            U32 const testLog = FUZ_rand(&lseed) % maxSrcLog;
            U32 const dictLog = FUZ_rand(&lseed) % maxSrcLog;
            U32 const cLevelCandidate = (FUZ_rand(&lseed) %
                               (ZSTD_maxCLevel() -
                               (MAX(testLog, dictLog) / 3))) +
                               1;
            U32 const cLevel = MIN(cLevelCandidate, cLevelMax);
            maxTestSize = FUZ_rLogLength(&lseed, testLog);
            oldTestLog = testLog;
            /* random dictionary selection */
            dictSize  = ((FUZ_rand(&lseed)&63)==1) ? FUZ_rLogLength(&lseed, dictLog) : 0;
            {   size_t const dictStart = FUZ_rand(&lseed) % (srcBufferSize - dictSize);
                dict = srcBuffer + dictStart;
                if (!dictSize) dict=NULL;
            }
            {   U64 const pledgedSrcSize = (FUZ_rand(&lseed) & 3) ? ZSTD_CONTENTSIZE_UNKNOWN : maxTestSize;
                ZSTD_compressionParameters cParams = ZSTD_getCParams(cLevel, pledgedSrcSize, dictSize);

                /* mess with compression parameters */
                cParams.windowLog += (FUZ_rand(&lseed) & 3) - 1;
                cParams.windowLog = MIN(windowLogMax, cParams.windowLog);
                cParams.hashLog += (FUZ_rand(&lseed) & 3) - 1;
                cParams.chainLog += (FUZ_rand(&lseed) & 3) - 1;
                cParams.searchLog += (FUZ_rand(&lseed) & 3) - 1;
                cParams.searchLength += (FUZ_rand(&lseed) & 3) - 1;
                cParams.targetLength = (U32)(cParams.targetLength * (0.5 + ((double)(FUZ_rand(&lseed) & 127) / 128)));
                cParams = ZSTD_adjustCParams(cParams, 0, 0);

                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_windowLog, cParams.windowLog, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_hashLog, cParams.hashLog, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_chainLog, cParams.chainLog, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_searchLog, cParams.searchLog, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_minMatch, cParams.searchLength, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_targetLength, cParams.targetLength, useOpaqueAPI) );

                /* mess with long distance matching parameters */
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_enableLongDistanceMatching, FUZ_rand(&lseed) & 63, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 3) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_ldmHashLog, FUZ_randomClampedLength(&lseed, ZSTD_HASHLOG_MIN, 23), useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 3) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_ldmMinMatch, FUZ_randomClampedLength(&lseed, ZSTD_LDM_MINMATCH_MIN, ZSTD_LDM_MINMATCH_MAX), useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 3) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_ldmBucketSizeLog, FUZ_randomClampedLength(&lseed, 0, ZSTD_LDM_BUCKETSIZELOG_MAX), useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 3) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_ldmHashEveryLog, FUZ_randomClampedLength(&lseed, 0, ZSTD_WINDOWLOG_MAX - ZSTD_HASHLOG_MIN), useOpaqueAPI) );

                /* unconditionally set, to be sync with decoder */
                /* mess with frame parameters */
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_checksumFlag, FUZ_rand(&lseed) & 1, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_dictIDFlag, FUZ_rand(&lseed) & 1, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_contentSizeFlag, FUZ_rand(&lseed) & 1, useOpaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( ZSTD_CCtx_setPledgedSrcSize(zc, pledgedSrcSize) );
                DISPLAYLEVEL(5, "pledgedSrcSize : %u \n", (U32)pledgedSrcSize);

                /* multi-threading parameters */
                {   U32 const nbThreadsCandidate = (FUZ_rand(&lseed) & 4) + 1;
                    U32 const nbThreads = MIN(nbThreadsCandidate, nbThreadsMax);
                    CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_nbThreads, nbThreads, useOpaqueAPI) );
                    if (nbThreads > 1) {
                        U32 const jobLog = FUZ_rand(&lseed) % (testLog+1);
                        CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_overlapSizeLog, FUZ_rand(&lseed) % 10, useOpaqueAPI) );
                        CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_jobSize, (U32)FUZ_rLogLength(&lseed, jobLog), useOpaqueAPI) );
                    }
                }

                if (FUZ_rand(&lseed) & 1) CHECK_Z (setCCtxParameter(zc, cctxParams, ZSTD_p_forceMaxWindow, FUZ_rand(&lseed) & 1, useOpaqueAPI) );

                /* Apply parameters */
                if (useOpaqueAPI) {
                    CHECK_Z (ZSTD_CCtx_setParametersUsingCCtxParams(zc, cctxParams) );
                }

                if (FUZ_rand(&lseed) & 1) {
                    if (FUZ_rand(&lseed) & 1) {
                        CHECK_Z( ZSTD_CCtx_loadDictionary(zc, dict, dictSize) );
                    } else {
                        CHECK_Z( ZSTD_CCtx_loadDictionary_byReference(zc, dict, dictSize) );
                    }
                    if (dict && dictSize) {
                        /* test that compression parameters are rejected (correctly) after loading a non-NULL dictionary */
                        if (useOpaqueAPI) {
                            size_t const setError = ZSTD_CCtx_setParametersUsingCCtxParams(zc, cctxParams);
                            CHECK(!ZSTD_isError(setError), "ZSTD_CCtx_setParametersUsingCCtxParams should have failed");
                        } else {
                            size_t const setError = ZSTD_CCtx_setParameter(zc, ZSTD_p_windowLog, cParams.windowLog-1);
                            CHECK(!ZSTD_isError(setError), "ZSTD_CCtx_setParameter should have failed");
                        }
                    }
                } else {
                    CHECK_Z( ZSTD_CCtx_refPrefix(zc, dict, dictSize) );
                }
        }   }

        /* multi-segments compression test */
        XXH64_reset(&xxhState, 0);
        {   ZSTD_outBuffer outBuff = { cBuffer, cBufferSize, 0 } ;
            for (cSize=0, totalTestSize=0 ; (totalTestSize < maxTestSize) ; ) {
                /* compress random chunks into randomly sized dst buffers */
                size_t const randomSrcSize = FUZ_randomLength(&lseed, maxSampleLog);
                size_t const srcSize = MIN(maxTestSize-totalTestSize, randomSrcSize);
                size_t const srcStart = FUZ_rand(&lseed) % (srcBufferSize - srcSize);
                size_t const randomDstSize = FUZ_randomLength(&lseed, maxSampleLog+1);
                size_t const dstBuffSize = MIN(cBufferSize - cSize, randomDstSize);
                ZSTD_EndDirective const flush = (FUZ_rand(&lseed) & 15) ? ZSTD_e_continue : ZSTD_e_flush;
                ZSTD_inBuffer inBuff = { srcBuffer+srcStart, srcSize, 0 };
                outBuff.size = outBuff.pos + dstBuffSize;

                CHECK_Z( ZSTD_compress_generic(zc, &outBuff, &inBuff, flush) );
                DISPLAYLEVEL(5, "compress consumed %u bytes (total : %u) \n",
                    (U32)inBuff.pos, (U32)(totalTestSize + inBuff.pos));

                XXH64_update(&xxhState, srcBuffer+srcStart, inBuff.pos);
                memcpy(copyBuffer+totalTestSize, srcBuffer+srcStart, inBuff.pos);
                totalTestSize += inBuff.pos;
            }

            /* final frame epilogue */
            {   size_t remainingToFlush = (size_t)(-1);
                while (remainingToFlush) {
                    ZSTD_inBuffer inBuff = { NULL, 0, 0 };
                    size_t const randomDstSize = FUZ_randomLength(&lseed, maxSampleLog+1);
                    size_t const adjustedDstSize = MIN(cBufferSize - cSize, randomDstSize);
                    outBuff.size = outBuff.pos + adjustedDstSize;
                    DISPLAYLEVEL(5, "End-flush into dst buffer of size %u \n", (U32)adjustedDstSize);
                    remainingToFlush = ZSTD_compress_generic(zc, &outBuff, &inBuff, ZSTD_e_end);
                    CHECK(ZSTD_isError(remainingToFlush),
                        "ZSTD_compress_generic w/ ZSTD_e_end error : %s",
                        ZSTD_getErrorName(remainingToFlush) );
            }   }
            crcOrig = XXH64_digest(&xxhState);
            cSize = outBuff.pos;
            DISPLAYLEVEL(5, "Frame completed : %u bytes \n", (U32)cSize);
        }

        /* multi - fragments decompression test */
        if (!dictSize /* don't reset if dictionary : could be different */ && (FUZ_rand(&lseed) & 1)) {
            DISPLAYLEVEL(5, "resetting DCtx (dict:%08X) \n", (U32)(size_t)dict);
            CHECK_Z( ZSTD_resetDStream(zd) );
        } else {
            DISPLAYLEVEL(5, "using dict of size %u \n", (U32)dictSize);
            CHECK_Z( ZSTD_initDStream_usingDict(zd, dict, dictSize) );
        }
        {   size_t decompressionResult = 1;
            ZSTD_inBuffer  inBuff = { cBuffer, cSize, 0 };
            ZSTD_outBuffer outBuff= { dstBuffer, dstBufferSize, 0 };
            for (totalGenSize = 0 ; decompressionResult ; ) {
                size_t const readCSrcSize = FUZ_randomLength(&lseed, maxSampleLog);
                size_t const randomDstSize = FUZ_randomLength(&lseed, maxSampleLog);
                size_t const dstBuffSize = MIN(dstBufferSize - totalGenSize, randomDstSize);
                inBuff.size = inBuff.pos + readCSrcSize;
                outBuff.size = inBuff.pos + dstBuffSize;
                DISPLAYLEVEL(5, "ZSTD_decompressStream input %u bytes (pos:%u/%u)\n",
                            (U32)readCSrcSize, (U32)inBuff.pos, (U32)cSize);
                decompressionResult = ZSTD_decompressStream(zd, &outBuff, &inBuff);
                CHECK (ZSTD_isError(decompressionResult), "decompression error : %s", ZSTD_getErrorName(decompressionResult));
                DISPLAYLEVEL(5, "inBuff.pos = %u \n", (U32)readCSrcSize);
            }
            CHECK (outBuff.pos != totalTestSize, "decompressed data : wrong size (%u != %u)", (U32)outBuff.pos, (U32)totalTestSize);
            CHECK (inBuff.pos != cSize, "compressed data should be fully read (%u != %u)", (U32)inBuff.pos, (U32)cSize);
            {   U64 const crcDest = XXH64(dstBuffer, totalTestSize, 0);
                if (crcDest!=crcOrig) findDiff(copyBuffer, dstBuffer, totalTestSize);
                CHECK (crcDest!=crcOrig, "decompressed data corrupted");
        }   }

        /*=====   noisy/erroneous src decompression test   =====*/

        /* add some noise */
        {   U32 const nbNoiseChunks = (FUZ_rand(&lseed) & 7) + 2;
            U32 nn; for (nn=0; nn<nbNoiseChunks; nn++) {
                size_t const randomNoiseSize = FUZ_randomLength(&lseed, maxSampleLog);
                size_t const noiseSize  = MIN((cSize/3) , randomNoiseSize);
                size_t const noiseStart = FUZ_rand(&lseed) % (srcBufferSize - noiseSize);
                size_t const cStart = FUZ_rand(&lseed) % (cSize - noiseSize);
                memcpy(cBuffer+cStart, srcBuffer+noiseStart, noiseSize);
        }   }

        /* try decompression on noisy data */
        CHECK_Z( ZSTD_initDStream(zd_noise) );   /* note : no dictionary */
        {   ZSTD_inBuffer  inBuff = { cBuffer, cSize, 0 };
            ZSTD_outBuffer outBuff= { dstBuffer, dstBufferSize, 0 };
            while (outBuff.pos < dstBufferSize) {
                size_t const randomCSrcSize = FUZ_randomLength(&lseed, maxSampleLog);
                size_t const randomDstSize = FUZ_randomLength(&lseed, maxSampleLog);
                size_t const adjustedDstSize = MIN(dstBufferSize - outBuff.pos, randomDstSize);
