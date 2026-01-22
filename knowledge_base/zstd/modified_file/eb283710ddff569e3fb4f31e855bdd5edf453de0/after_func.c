                    int gain1 = (int)(matchLength*4 - ZSTD_highbit((U32)offset+1) + 7);
                    if (gain2 > gain1)
                    {
                        matchLength = ml2, offset = offset2, start = ip;
                        continue;
                    }
                }
            }
            break;  /* nothing found : store previous solution */
        }

        /* store sequence */
        {
            size_t litLength = start - anchor;
            if (offset) offset_1 = offset;
            ZSTD_storeSeq(seqStorePtr, litLength, anchor, offset, matchLength-MINMATCH);
            ip = start + matchLength;
            anchor = ip;
        }

    }

    /* Last Literals */
    {
        size_t lastLLSize = iend - anchor;
        memcpy(seqStorePtr->lit, anchor, lastLLSize);
        seqStorePtr->lit += lastLLSize;
    }

    /* Final compression stage */
    return ZSTD_compressSequences((BYTE*)dst, maxDstSize,
                                  seqStorePtr, srcSize);
}

size_t ZSTD_HC_compressBlock_btlazy2(ZSTD_HC_CCtx* ctx, void* dst, size_t maxDstSize, const void* src, size_t srcSize)
{
    return ZSTD_HC_compressBlock_lazy_generic(ctx, dst, maxDstSize, src, srcSize, 1, 1);
}

size_t ZSTD_HC_compressBlock_lazy2(ZSTD_HC_CCtx* ctx, void* dst, size_t maxDstSize, const void* src, size_t srcSize)
{
    return ZSTD_HC_compressBlock_lazy_generic(ctx, dst, maxDstSize, src, srcSize, 0, 1);
}

size_t ZSTD_HC_compressBlock_lazy(ZSTD_HC_CCtx* ctx, void* dst, size_t maxDstSize, const void* src, size_t srcSize)
{
    return ZSTD_HC_compressBlock_lazy_generic(ctx, dst, maxDstSize, src, srcSize, 0, 0);
}


size_t ZSTD_HC_compressBlock_greedy(ZSTD_HC_CCtx* ctx, void* dst, size_t maxDstSize, const void* src, size_t srcSize)
{
    seqStore_t* seqStorePtr = &(ctx->seqStore);
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;

    size_t offset_2=REPCODE_STARTVALUE, offset_1=REPCODE_STARTVALUE;
    const U32 maxSearches = 1 << ctx->params.searchLog;
    const U32 mls = ctx->params.searchLength;

    /* init */
    ZSTD_resetSeqStore(seqStorePtr);
    if (((ip-ctx->base) - ctx->dictLimit) < REPCODE_STARTVALUE) ip += REPCODE_STARTVALUE;

    /* Match Loop */
    while (ip < ilimit)
    {
        /* repcode */
        if (MEM_read32(ip) == MEM_read32(ip - offset_2))
        {
            /* store sequence */
            size_t matchLength = ZSTD_count(ip+MINMATCH, ip+MINMATCH-offset_2, iend);
            size_t litLength = ip-anchor;
            size_t offset = offset_2;
            offset_2 = offset_1;
            offset_1 = offset;
            ZSTD_storeSeq(seqStorePtr, litLength, anchor, 0, matchLength);
            ip += matchLength+MINMATCH;
