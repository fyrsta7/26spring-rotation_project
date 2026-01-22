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
                               int useOpaqueAPI)
{
    if (useOpaqueAPI) {
        return ZSTD_CCtxParam_setParameter(cctxParams, param, value);
    } else {
        return ZSTD_CCtx_setParameter(zc, param, value);
    }
}

/* Tests for ZSTD_compress_generic() API */
static int fuzzerTests_newAPI(U32 seed, U32 nbTests, unsigned startTest,
                              double compressibility, int bigTests)
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
    UTIL_time_t const startClock = UTIL_getTime();
    const BYTE* dict = NULL;   /* can keep same dict on 2 consecutive tests */
    size_t dictSize = 0;
    U32 oldTestLog = 0;
    U32 windowLogMalus = 0;   /* can survive between 2 loops */
    U32 const cLevelMax = bigTests ? (U32)ZSTD_maxCLevel()-1 : g_cLevelMax_smallTests;
    U32 const nbThreadsMax = bigTests ? 4 : 2;
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
    for ( ; (testNb <= nbTests) || (UTIL_clockSpanMicro(startClock) < g_clockTime) ; testNb++ ) {
        U32 lseed;
        int opaqueAPI;
        const BYTE* srcBuffer;
        size_t totalTestSize, totalGenSize, cSize;
        XXH64_state_t xxhState;
        U64 crcOrig;
        U32 resetAllowed = 1;
        size_t maxTestSize;
        ZSTD_parameters savedParams;

        /* init */
        if (nbTests >= testNb) { DISPLAYUPDATE(2, "\r%6u/%6u    ", testNb, nbTests); }
        else { DISPLAYUPDATE(2, "\r%6u          ", testNb); }
        FUZ_rand(&coreSeed);
        lseed = coreSeed ^ prime32;
        DISPLAYLEVEL(5, " ***  Test %u  *** \n", testNb);
        opaqueAPI = FUZ_rand(&lseed) & 1;

        /* states full reset (deliberately not synchronized) */
        /* some issues can only happen when reusing states */
        if ((FUZ_rand(&lseed) & 0xFF) == 131) {
            DISPLAYLEVEL(5, "Creating new context \n");
            ZSTD_freeCCtx(zc);
            zc = ZSTD_createCCtx();
            CHECK(zc == NULL, "ZSTD_createCCtx allocation error");
            resetAllowed = 0;
        }
        if ((FUZ_rand(&lseed) & 0xFF) == 132) {
            ZSTD_freeDStream(zd);
            zd = ZSTD_createDStream();
            CHECK(zd == NULL, "ZSTD_createDStream allocation error");
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
          && oldTestLog   /* at least one test happened */
          && resetAllowed) {
            /* just set a compression level */
            maxTestSize = FUZ_randomLength(&lseed, oldTestLog+2);
            if (maxTestSize >= srcBufferSize) maxTestSize = srcBufferSize-1;
            {   int const compressionLevel = (FUZ_rand(&lseed) % 5) + 1;
                DISPLAYLEVEL(5, "t%u : compression level : %i \n", testNb, compressionLevel);
                CHECK_Z (setCCtxParameter(zc, cctxParams, ZSTD_p_compressionLevel, compressionLevel, opaqueAPI) );
            }
        } else {
            U32 const testLog = FUZ_rand(&lseed) % maxSrcLog;
            U32 const dictLog = FUZ_rand(&lseed) % maxSrcLog;
            U32 const cLevelCandidate = (FUZ_rand(&lseed) %
                               (ZSTD_maxCLevel() -
                               (MAX(testLog, dictLog) / 2))) +
                               1;
            U32 const cLevel = MIN(cLevelCandidate, cLevelMax);
            DISPLAYLEVEL(5, "t%u: base cLevel : %u \n", testNb, cLevel);
            maxTestSize = FUZ_rLogLength(&lseed, testLog);
            DISPLAYLEVEL(5, "t%u: maxTestSize : %u \n", testNb, (U32)maxTestSize);
            oldTestLog = testLog;
            /* random dictionary selection */
            dictSize  = ((FUZ_rand(&lseed)&63)==1) ? FUZ_rLogLength(&lseed, dictLog) : 0;
            {   size_t const dictStart = FUZ_rand(&lseed) % (srcBufferSize - dictSize);
                dict = srcBuffer + dictStart;
                if (!dictSize) dict=NULL;
            }
            {   U64 const pledgedSrcSize = (FUZ_rand(&lseed) & 3) ? ZSTD_CONTENTSIZE_UNKNOWN : maxTestSize;
                ZSTD_compressionParameters cParams = ZSTD_getCParams(cLevel, pledgedSrcSize, dictSize);
                const U32 windowLogMax = bigTests ? 24 : 20;
                const U32 searchLogMax = bigTests ? 15 : 13;
                if (dictSize)
                    DISPLAYLEVEL(5, "t%u: with dictionary of size : %zu \n", testNb, dictSize);

                /* mess with compression parameters */
                cParams.windowLog += (FUZ_rand(&lseed) & 3) - 1;
                cParams.windowLog = MIN(windowLogMax, cParams.windowLog);
                cParams.hashLog += (FUZ_rand(&lseed) & 3) - 1;
                cParams.chainLog += (FUZ_rand(&lseed) & 3) - 1;
                cParams.searchLog += (FUZ_rand(&lseed) & 3) - 1;
                cParams.searchLog = MIN(searchLogMax, cParams.searchLog);
                cParams.searchLength += (FUZ_rand(&lseed) & 3) - 1;
                cParams.targetLength = (U32)((cParams.targetLength + 1 ) * (0.5 + ((double)(FUZ_rand(&lseed) & 127) / 128)));
                cParams = ZSTD_adjustCParams(cParams, pledgedSrcSize, dictSize);

                if (FUZ_rand(&lseed) & 1) {
                    DISPLAYLEVEL(5, "t%u: windowLog : %u \n", testNb, cParams.windowLog);
                    CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_windowLog, cParams.windowLog, opaqueAPI) );
                    assert(cParams.windowLog >= ZSTD_WINDOWLOG_MIN);   /* guaranteed by ZSTD_adjustCParams() */
                    windowLogMalus = (cParams.windowLog - ZSTD_WINDOWLOG_MIN) / 5;
                }
                if (FUZ_rand(&lseed) & 1) {
                    DISPLAYLEVEL(5, "t%u: hashLog : %u \n", testNb, cParams.hashLog);
                    CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_hashLog, cParams.hashLog, opaqueAPI) );
                }
                if (FUZ_rand(&lseed) & 1) {
                    DISPLAYLEVEL(5, "t%u: chainLog : %u \n", testNb, cParams.chainLog);
                    CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_chainLog, cParams.chainLog, opaqueAPI) );
                }
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_searchLog, cParams.searchLog, opaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_minMatch, cParams.searchLength, opaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_targetLength, cParams.targetLength, opaqueAPI) );

                /* mess with long distance matching parameters */
                if (bigTests) {
                    if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_enableLongDistanceMatching, FUZ_rand(&lseed) & 63, opaqueAPI) );
                    if (FUZ_rand(&lseed) & 3) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_ldmHashLog, FUZ_randomClampedLength(&lseed, ZSTD_HASHLOG_MIN, 23), opaqueAPI) );
                    if (FUZ_rand(&lseed) & 3) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_ldmMinMatch, FUZ_randomClampedLength(&lseed, ZSTD_LDM_MINMATCH_MIN, ZSTD_LDM_MINMATCH_MAX), opaqueAPI) );
                    if (FUZ_rand(&lseed) & 3) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_ldmBucketSizeLog, FUZ_randomClampedLength(&lseed, 0, ZSTD_LDM_BUCKETSIZELOG_MAX), opaqueAPI) );
                    if (FUZ_rand(&lseed) & 3) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_ldmHashEveryLog, FUZ_randomClampedLength(&lseed, 0, ZSTD_WINDOWLOG_MAX - ZSTD_HASHLOG_MIN), opaqueAPI) );
                }

                /* mess with frame parameters */
                if (FUZ_rand(&lseed) & 1) {
                    U32 const checksumFlag = FUZ_rand(&lseed) & 1;
                    DISPLAYLEVEL(5, "t%u: frame checksum : %u \n", testNb, checksumFlag);
                    CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_checksumFlag, checksumFlag, opaqueAPI) );
                }
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_dictIDFlag, FUZ_rand(&lseed) & 1, opaqueAPI) );
                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_contentSizeFlag, FUZ_rand(&lseed) & 1, opaqueAPI) );
                if (FUZ_rand(&lseed) & 1) {
                    DISPLAYLEVEL(5, "t%u: pledgedSrcSize : %u \n", testNb, (U32)pledgedSrcSize);
                    CHECK_Z( ZSTD_CCtx_setPledgedSrcSize(zc, pledgedSrcSize) );
                }

                /* multi-threading parameters. Only adjust ocassionally for small tests. */
                if (bigTests || (FUZ_rand(&lseed) & 0xF) == 0xF) {
                    U32 const nbThreadsCandidate = (FUZ_rand(&lseed) & 4) + 1;
                    U32 const nbThreadsAdjusted = (windowLogMalus < nbThreadsCandidate) ? nbThreadsCandidate - windowLogMalus : 1;
                    U32 const nbThreads = MIN(nbThreadsAdjusted, nbThreadsMax);
                    DISPLAYLEVEL(5, "t%u: nbThreads : %u \n", testNb, nbThreads);
                    CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_nbWorkers, nbThreads, opaqueAPI) );
                    if (nbThreads > 1) {
                        U32 const jobLog = FUZ_rand(&lseed) % (testLog+1);
                        CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_overlapSizeLog, FUZ_rand(&lseed) % 10, opaqueAPI) );
                        CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_jobSize, (U32)FUZ_rLogLength(&lseed, jobLog), opaqueAPI) );
                    }
                }

                if (FUZ_rand(&lseed) & 1) CHECK_Z( setCCtxParameter(zc, cctxParams, ZSTD_p_forceMaxWindow, FUZ_rand(&lseed) & 1, opaqueAPI) );

                /* Apply parameters */
                if (opaqueAPI) {
                    DISPLAYLEVEL(5, "t%u: applying CCtxParams \n", testNb);
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
                        if (opaqueAPI) {
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

        CHECK_Z(getCCtxParams(zc, &savedParams));

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
                DISPLAYLEVEL(6, "t%u: compress consumed %u bytes (total : %u) ; flush: %u (total : %u) \n",
                    testNb, (U32)inBuff.pos, (U32)(totalTestSize + inBuff.pos), (U32)flush, (U32)outBuff.pos);

                XXH64_update(&xxhState, srcBuffer+srcStart, inBuff.pos);
                memcpy(copyBuffer+totalTestSize, srcBuffer+srcStart, inBuff.pos);
                totalTestSize += inBuff.pos;
            }

            /* final frame epilogue */
            {   size_t remainingToFlush = 1;
                while (remainingToFlush) {
                    ZSTD_inBuffer inBuff = { NULL, 0, 0 };
                    size_t const randomDstSize = FUZ_randomLength(&lseed, maxSampleLog+1);
                    size_t const adjustedDstSize = MIN(cBufferSize - cSize, randomDstSize);
                    outBuff.size = outBuff.pos + adjustedDstSize;
                    DISPLAYLEVEL(6, "t%u: End-flush into dst buffer of size %u \n", testNb, (U32)adjustedDstSize);
                    remainingToFlush = ZSTD_compress_generic(zc, &outBuff, &inBuff, ZSTD_e_end);
                    DISPLAYLEVEL(6, "t%u: Total flushed so far : %u bytes \n", testNb, (U32)outBuff.pos);
                    CHECK( ZSTD_isError(remainingToFlush),
                          "ZSTD_compress_generic w/ ZSTD_e_end error : %s",
                           ZSTD_getErrorName(remainingToFlush) );
            }   }
            crcOrig = XXH64_digest(&xxhState);
            cSize = outBuff.pos;
            DISPLAYLEVEL(5, "Frame completed : %zu bytes \n", cSize);
        }

        CHECK(badParameters(zc, savedParams), "CCtx params are wrong");

        /* multi - fragments decompression test */
        if (!dictSize /* don't reset if dictionary : could be different */ && (FUZ_rand(&lseed) & 1)) {
            DISPLAYLEVEL(5, "resetting DCtx (dict:%08X) \n", (U32)(size_t)dict);
            CHECK_Z( ZSTD_resetDStream(zd) );
        } else {
            if (dictSize)
                DISPLAYLEVEL(5, "using dictionary of size %zu \n", dictSize);
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
                outBuff.size = outBuff.pos + dstBuffSize;
                DISPLAYLEVEL(6, "decompression presented %u new bytes (pos:%u/%u)\n",
                                (U32)readCSrcSize, (U32)inBuff.pos, (U32)cSize);
                decompressionResult = ZSTD_decompressStream(zd, &outBuff, &inBuff);
                DISPLAYLEVEL(6, "so far: consumed = %u, produced = %u \n",
                                (U32)inBuff.pos, (U32)outBuff.pos);
                if (ZSTD_isError(decompressionResult)) {
                    DISPLAY("ZSTD_decompressStream error : %s \n", ZSTD_getErrorName(decompressionResult));
                    findDiff(copyBuffer, dstBuffer, totalTestSize);
                }
                CHECK (ZSTD_isError(decompressionResult), "decompression error : %s", ZSTD_getErrorName(decompressionResult));
                CHECK (inBuff.pos > cSize, "ZSTD_decompressStream consumes too much input : %u > %u ", (U32)inBuff.pos, (U32)cSize);
            }
            CHECK (inBuff.pos != cSize, "compressed data should be fully read (%u != %u)", (U32)inBuff.pos, (U32)cSize);
            CHECK (outBuff.pos != totalTestSize, "decompressed data : wrong size (%u != %u)", (U32)outBuff.pos, (U32)totalTestSize);
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
