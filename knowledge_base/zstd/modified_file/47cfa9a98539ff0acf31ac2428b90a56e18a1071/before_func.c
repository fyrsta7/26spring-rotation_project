            {
                const size_t mlt = ZSTD_count(ip+MINMATCH, match+MINMATCH, iLimit) + MINMATCH;
                if (mlt > ml)
                //if (((int)(4*mlt) - (int)ZSTD_highbit((U32)(ip-match)+1)) > ((int)(4*ml) - (int)ZSTD_highbit((U32)((*offsetPtr)+1))))
                {
                    ml = mlt; *offsetPtr = ip-match;
                    if (ip+ml >= iLimit) break;
                }
            }
        }
        else
        {
            match = dictBase + matchIndex;
            if (MEM_read32(match) == MEM_read32(ip))
            {
                size_t mlt;
                const BYTE* vLimit = ip + (dictLimit - matchIndex);
                if (vLimit > iLimit) vLimit = iLimit;
                mlt = ZSTD_count(ip+MINMATCH, match+MINMATCH, vLimit) + MINMATCH;
                if ((ip+mlt == vLimit) && (vLimit < iLimit))
                    mlt += ZSTD_count(ip+mlt, base+dictLimit, iLimit);
                if (mlt > ml) { ml = mlt; *offsetPtr = (ip-base) - matchIndex; }
            }
        }

        if (base + matchIndex <= ip - chainSize) break;
        matchIndex = NEXT_IN_CHAIN(matchIndex, chainMask);
    }

    return ml;
}


FORCE_INLINE size_t ZSTD_HC_HcFindBestMatch_selectMLS (
                        ZSTD_HC_CCtx* zc,   /* Index table will be updated */
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 maxNbAttempts, const U32 matchLengthSearch)
{
    switch(matchLengthSearch)
    {
    default :
    case 4 : return ZSTD_HC_HcFindBestMatch(zc, ip, iLimit, offsetPtr, maxNbAttempts, 4);
    case 5 : return ZSTD_HC_HcFindBestMatch(zc, ip, iLimit, offsetPtr, maxNbAttempts, 5);
    case 6 : return ZSTD_HC_HcFindBestMatch(zc, ip, iLimit, offsetPtr, maxNbAttempts, 6);
    }
}


/* common lazy function, to be inlined */
FORCE_INLINE
size_t ZSTD_HC_compressBlock_lazy_generic(ZSTD_HC_CCtx* ctx,
                                     void* dst, size_t maxDstSize, const void* src, size_t srcSize,
                                     const U32 searchMethod, const U32 deep)   /* 0 : hc; 1 : bt */
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

    typedef size_t (*searchMax_f)(ZSTD_HC_CCtx* zc, const BYTE* ip, const BYTE* iLimit,
                        size_t* offsetPtr,
                        U32 maxNbAttempts, U32 matchLengthSearch);
    searchMax_f searchMax = searchMethod ? ZSTD_HC_BtFindBestMatch_selectMLS : ZSTD_HC_HcFindBestMatch_selectMLS;

    /* init */
    ZSTD_resetSeqStore(seqStorePtr);
    if (((ip-ctx->base) - ctx->dictLimit) < REPCODE_STARTVALUE) ip += REPCODE_STARTVALUE;

    /* Match Loop */
    while (ip <= ilimit)
    {
        size_t matchLength;
        size_t offset=999999;
        const BYTE* start;

        /* try to find a first match */
        if (MEM_read32(ip) == MEM_read32(ip - offset_2))
        {
            /* repcode : we take it*/
            size_t offtmp = offset_2;
            size_t litLength = ip - anchor;
            matchLength = ZSTD_count(ip+MINMATCH, ip+MINMATCH-offset_2, iend);
            offset_2 = offset_1;
            offset_1 = offtmp;
            ZSTD_storeSeq(seqStorePtr, litLength, anchor, 0, matchLength);
            ip += matchLength+MINMATCH;
            anchor = ip;
            continue;
        }

        offset_2 = offset_1;
        matchLength = searchMax(ctx, ip, iend, &offset, maxSearches, mls);
        if (!matchLength)
        {
            ip += ((ip-anchor) >> g_searchStrength) + 1;   /* jump faster over incompressible sections */
            continue;
        }

        /* let's try to find a better solution */
        start = ip;

        while (ip<ilimit)
        {
            ip ++;
            if (MEM_read32(ip) == MEM_read32(ip - offset_1))
            {
                size_t ml2 = ZSTD_count(ip+MINMATCH, ip+MINMATCH-offset_1, iend) + MINMATCH;
                int gain2 = (int)(ml2 * 3);
                int gain1 = (int)(matchLength*3 - ZSTD_highbit((U32)offset+1) + 1);
                if (gain2 > gain1)
                    matchLength = ml2, offset = 0, start = ip;
            }
            {
                size_t offset2=999999;
                size_t ml2 = searchMax(ctx, ip, iend, &offset2, maxSearches, mls);
                int gain2 = (int)(ml2*(3+deep) - ZSTD_highbit((U32)offset2+1));   /* raw approx */
                int gain1 = (int)(matchLength*(3+deep) - ZSTD_highbit((U32)offset+1) + (3+deep));
                if (gain2 > gain1)
                {
                    matchLength = ml2, offset = offset2, start = ip;
                    continue;   /* search a better one */
                }
            }
