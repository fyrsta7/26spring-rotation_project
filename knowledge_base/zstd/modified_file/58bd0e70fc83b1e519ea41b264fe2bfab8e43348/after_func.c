        return ERROR(compressionParameter_unsupported);
    }
}


/* ------------------------------------------ */
/* =====   Multi-threaded compression   ===== */
/* ------------------------------------------ */

size_t ZSTDMT_compress_advanced(ZSTDMT_CCtx* mtctx,
                           void* dst, size_t dstCapacity,
                     const void* src, size_t srcSize,
                     const ZSTD_CDict* cdict,
                           ZSTD_parameters const params,
                           unsigned overlapRLog)
{
    size_t const overlapSize = (overlapRLog>=9) ? 0 : (size_t)1 << (params.cParams.windowLog - overlapRLog);
    size_t const chunkSizeTarget = (size_t)1 << (params.cParams.windowLog + 2);
    size_t const chunkMaxSize = chunkSizeTarget << 2;
    size_t const passSizeMax = chunkMaxSize * mtctx->nbThreads;
    unsigned const multiplier = (unsigned)(srcSize / passSizeMax) + 1;
    unsigned nbChunksLarge = multiplier * mtctx->nbThreads;
    unsigned const nbChunksMax = (unsigned)(srcSize / chunkSizeTarget) + 1;
    unsigned nbChunksSmall = MIN(nbChunksMax, mtctx->nbThreads);
    unsigned nbChunks = (multiplier>1) ? nbChunksLarge : nbChunksSmall;
    size_t const proposedChunkSize = (srcSize + (nbChunks-1)) / nbChunks;
    size_t const avgChunkSize = ((proposedChunkSize & 0x1FFFF) < 0x7FFF) ? proposedChunkSize + 0xFFFF : proposedChunkSize;   /* avoid too small last block */
    const char* const srcStart = (const char*)src;
    size_t remainingSrcSize = srcSize;
    unsigned const compressWithinDst = (dstCapacity >= ZSTD_compressBound(srcSize)) ? nbChunks : (unsigned)(dstCapacity / ZSTD_compressBound(avgChunkSize));  /* presumes avgChunkSize >= 256 KB, which should be the case */
    size_t frameStartPos = 0, dstBufferPos = 0;

    DEBUGLOG(4, "windowLog : %2u => chunkSizeTarget : %u bytes  ", params.cParams.windowLog, (U32)chunkSizeTarget);
    DEBUGLOG(4, "nbChunks  : %2u   (chunkSize : %u bytes)   ", nbChunks, (U32)avgChunkSize);
    assert(avgChunkSize >= 256 KB);  /* required for ZSTD_compressBound(A) + ZSTD_compressBound(B) <= ZSTD_compressBound(A+B) */

    if (nbChunks==1) {   /* fallback to single-thread mode */
        ZSTD_CCtx* const cctx = mtctx->cctxPool->cctx[0];
        if (cdict) return ZSTD_compress_usingCDict_advanced(cctx, dst, dstCapacity, src, srcSize, cdict, params.fParams);
        return ZSTD_compress_advanced(cctx, dst, dstCapacity, src, srcSize, NULL, 0, params);
    }

    if (nbChunks > mtctx->jobIDMask+1) {  /* enlarge job table */
        U32 nbJobs = nbChunks;
        ZSTD_free(mtctx->jobs, mtctx->cMem);
        mtctx->jobIDMask = 0;
        mtctx->jobs = ZSTDMT_allocJobsTable(&nbJobs, mtctx->cMem);
        if (mtctx->jobs==NULL) return ERROR(memory_allocation);
        mtctx->jobIDMask = nbJobs - 1;
    }

    {   unsigned u;
        for (u=0; u<nbChunks; u++) {
            size_t const chunkSize = MIN(remainingSrcSize, avgChunkSize);
            size_t const dstBufferCapacity = ZSTD_compressBound(chunkSize);
            buffer_t const dstAsBuffer = { (char*)dst + dstBufferPos, dstBufferCapacity };
            buffer_t const dstBuffer = u < compressWithinDst ? dstAsBuffer : ZSTDMT_getBuffer(mtctx->buffPool, dstBufferCapacity);
            ZSTD_CCtx* const cctx = ZSTDMT_getCCtx(mtctx->cctxPool);
            size_t dictSize = u ? overlapSize : 0;

            if ((cctx==NULL) || (dstBuffer.start==NULL)) {
                mtctx->jobs[u].cSize = ERROR(memory_allocation);   /* job result */
                mtctx->jobs[u].jobCompleted = 1;
                nbChunks = u+1;
                break;   /* let's wait for previous jobs to complete, but don't start new ones */
            }

            mtctx->jobs[u].srcStart = srcStart + frameStartPos - dictSize;
            mtctx->jobs[u].dictSize = dictSize;
            mtctx->jobs[u].srcSize = chunkSize;
            mtctx->jobs[u].cdict = mtctx->nextJobID==0 ? cdict : NULL;
            mtctx->jobs[u].fullFrameSize = srcSize;
            mtctx->jobs[u].params = params;
            /* do not calculate checksum within sections, but write it in header for first section */
            if (u!=0) mtctx->jobs[u].params.fParams.checksumFlag = 0;
            mtctx->jobs[u].dstBuff = dstBuffer;
            mtctx->jobs[u].cctx = cctx;
            mtctx->jobs[u].firstChunk = (u==0);
            mtctx->jobs[u].lastChunk = (u==nbChunks-1);
            mtctx->jobs[u].jobCompleted = 0;
            mtctx->jobs[u].jobCompleted_mutex = &mtctx->jobCompleted_mutex;
            mtctx->jobs[u].jobCompleted_cond = &mtctx->jobCompleted_cond;

            DEBUGLOG(5, "posting job %u   (%u bytes)", u, (U32)chunkSize);
            DEBUG_PRINTHEX(6, mtctx->jobs[u].srcStart, 12);
            POOL_add(mtctx->factory, ZSTDMT_compressChunk, &mtctx->jobs[u]);

            frameStartPos += chunkSize;
            dstBufferPos += dstBufferCapacity;
            remainingSrcSize -= chunkSize;
    }   }
    /* note : since nbChunks <= nbThreads, all jobs should be running immediately in parallel */

    {   unsigned chunkID;
        size_t error = 0, dstPos = 0;
        for (chunkID=0; chunkID<nbChunks; chunkID++) {
            DEBUGLOG(5, "waiting for chunk %u ", chunkID);
            PTHREAD_MUTEX_LOCK(&mtctx->jobCompleted_mutex);
            while (mtctx->jobs[chunkID].jobCompleted==0) {
                DEBUGLOG(5, "waiting for jobCompleted signal from chunk %u", chunkID);
                pthread_cond_wait(&mtctx->jobCompleted_cond, &mtctx->jobCompleted_mutex);
            }
            pthread_mutex_unlock(&mtctx->jobCompleted_mutex);
            DEBUGLOG(5, "ready to write chunk %u ", chunkID);

            ZSTDMT_releaseCCtx(mtctx->cctxPool, mtctx->jobs[chunkID].cctx);
            mtctx->jobs[chunkID].cctx = NULL;
            mtctx->jobs[chunkID].srcStart = NULL;
            {   size_t const cSize = mtctx->jobs[chunkID].cSize;
                if (ZSTD_isError(cSize)) error = cSize;
                if ((!error) && (dstPos + cSize > dstCapacity)) error = ERROR(dstSize_tooSmall);
                if (chunkID) {   /* note : chunk 0 is already written directly into dst */
                    if (!error)
                        memmove((char*)dst + dstPos, mtctx->jobs[chunkID].dstBuff.start, cSize);  /* may overlap if chunk decompressed within dst */
                    if (chunkID >= compressWithinDst) {  /* otherwise, it decompresses within dst */
                        DEBUGLOG(5, "releasing buffer %u>=%u", chunkID, compressWithinDst);
                        ZSTDMT_releaseBuffer(mtctx->buffPool, mtctx->jobs[chunkID].dstBuff);
                    }
