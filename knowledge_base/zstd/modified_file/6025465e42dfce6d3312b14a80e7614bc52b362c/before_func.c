    ZSTD_pthread_mutex_t* jobCompleted_mutex;
    ZSTD_pthread_cond_t* jobCompleted_cond;
    ZSTD_CCtx_params params;
    const ZSTD_CDict* cdict;
    ZSTDMT_CCtxPool* cctxPool;
    ZSTDMT_bufferPool* bufPool;
    unsigned long long fullFrameSize;
} ZSTDMT_jobDescription;

/* ZSTDMT_compressChunk() is a POOL_function type */
void ZSTDMT_compressChunk(void* jobDescription)
{
    ZSTDMT_jobDescription* const job = (ZSTDMT_jobDescription*)jobDescription;
    ZSTD_CCtx* const cctx = ZSTDMT_getCCtx(job->cctxPool);
    const void* const src = (const char*)job->srcStart + job->prefixSize;
    buffer_t dstBuff = job->dstBuff;

    /* ressources */
    if (cctx==NULL) {
        job->cSize = ERROR(memory_allocation);
        goto _endJob;
    }
    if (dstBuff.start == NULL) {
        dstBuff = ZSTDMT_getBuffer(job->bufPool);
        if (dstBuff.start==NULL) {
            job->cSize = ERROR(memory_allocation);
            goto _endJob;
        }
        job->dstBuff = dstBuff;
    }

    /* init */
    if (job->cdict) {
        size_t const initError = ZSTD_compressBegin_advanced_internal(cctx, NULL, 0, ZSTD_dm_auto, job->cdict, job->params, job->fullFrameSize);
        assert(job->firstChunk);  /* only allowed for first job */
        if (ZSTD_isError(initError)) { job->cSize = initError; goto _endJob; }
    } else {  /* srcStart points at reloaded section */
        U64 const pledgedSrcSize = job->firstChunk ? job->fullFrameSize : ZSTD_CONTENTSIZE_UNKNOWN;
        ZSTD_CCtx_params jobParams = job->params;   /* do not modify job->params ! copy it, modify the copy */
        {   size_t const forceWindowError = ZSTD_CCtxParam_setParameter(&jobParams, ZSTD_p_forceMaxWindow, !job->firstChunk);
            if (ZSTD_isError(forceWindowError)) {
                job->cSize = forceWindowError;
                goto _endJob;
        }   }
        {   size_t const initError = ZSTD_compressBegin_advanced_internal(cctx,
                                        job->srcStart, job->prefixSize, ZSTD_dm_rawContent, /* load dictionary in "content-only" mode (no header analysis) */
                                        NULL,
                                        jobParams, pledgedSrcSize);
            if (ZSTD_isError(initError)) {
                job->cSize = initError;
                goto _endJob;
        }   }
    }
    if (!job->firstChunk) {  /* flush and overwrite frame header when it's not first job */
        size_t const hSize = ZSTD_compressContinue(cctx, dstBuff.start, dstBuff.size, src, 0);
        if (ZSTD_isError(hSize)) { job->cSize = hSize; /* save error code */ goto _endJob; }
        DEBUGLOG(5, "ZSTDMT_compressChunk: flush and overwrite %u bytes of frame header (not first chunk)", (U32)hSize);
        ZSTD_invalidateRepCodes(cctx);
    }

    /* compress */
#if 0
    job->cSize = (job->lastChunk) ?
                 ZSTD_compressEnd     (cctx, dstBuff.start, dstBuff.size, src, job->srcSize) :
                 ZSTD_compressContinue(cctx, dstBuff.start, dstBuff.size, src, job->srcSize);
#else
    if (sizeof(size_t) > sizeof(int))
        assert(job->srcSize < ((size_t)INT_MAX) * ZSTD_BLOCKSIZE_MAX);   /* check overflow */
    {   int const nbBlocks = (int)((job->srcSize + (ZSTD_BLOCKSIZE_MAX-1)) / ZSTD_BLOCKSIZE_MAX);
        const BYTE* ip = (const BYTE*) src;
        BYTE* const ostart = (BYTE*)dstBuff.start;
        BYTE* op = ostart;
        BYTE* oend = op + dstBuff.size;
        int blockNb;
        DEBUGLOG(5, "ZSTDMT_compressChunk: compress %u bytes in %i blocks", (U32)job->srcSize, nbBlocks);
        job->cSize = 0;
        for (blockNb = 1; blockNb < nbBlocks; blockNb++) {
            size_t const cSize = ZSTD_compressContinue(cctx, op, oend-op, ip, ZSTD_BLOCKSIZE_MAX);
            if (ZSTD_isError(cSize)) { job->cSize = cSize; goto _endJob; }
            ip += ZSTD_BLOCKSIZE_MAX;
            op += cSize; assert(op < oend);
            /* stats */
            job->cSize += cSize;
            job->readSize = ZSTD_BLOCKSIZE_MAX * blockNb;
        }
        /* last block */
        if ((nbBlocks > 0) | job->lastChunk /*need to output a "last block" flag*/ ) {
            size_t const lastBlockSize1 = job->srcSize & (ZSTD_BLOCKSIZE_MAX-1);
            size_t const lastBlockSize = ((lastBlockSize1==0) & (job->srcSize>=ZSTD_BLOCKSIZE_MAX)) ? ZSTD_BLOCKSIZE_MAX : lastBlockSize1;
            size_t const cSize = (job->lastChunk) ?
                 ZSTD_compressEnd     (cctx, op, oend-op, ip, lastBlockSize) :
                 ZSTD_compressContinue(cctx, op, oend-op, ip, lastBlockSize);
            if (ZSTD_isError(cSize)) { job->cSize = cSize; goto _endJob; }
            /* stats */
            job->cSize += cSize;
            job->readSize = job->srcSize;
        }
    }
#endif

_endJob:
    /* release */
