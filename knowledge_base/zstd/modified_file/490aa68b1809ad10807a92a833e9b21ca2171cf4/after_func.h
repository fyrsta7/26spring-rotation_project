    }
}

/** Tree updater, providing best match */
FORCE_INLINE /* inlining is important to hardwire a hot branch (template emulation) */
U32 ZSTD_BtGetAllMatches_extDict (
                        ZSTD_CCtx* zc,
                        const BYTE* const ip, const BYTE* const iLimit,
                        const U32 maxNbAttempts, const U32 mls, ZSTD_match_t* matches, U32 minml)
{
    if (ip < zc->base + zc->nextToUpdate) return 0;   /* skipped area */
    ZSTD_updateTree_extDict(zc, ip, iLimit, maxNbAttempts, mls);
    return ZSTD_insertBtAndGetAllMatches(zc, ip, iLimit, maxNbAttempts, mls, 1, matches, minml);
}


FORCE_INLINE U32 ZSTD_BtGetAllMatches_selectMLS_extDict (
                        ZSTD_CCtx* zc,   /* Index table will be updated */
                        const BYTE* ip, const BYTE* const iLowLimit, const BYTE* const iHighLimit,
                        const U32 maxNbAttempts, const U32 matchLengthSearch, ZSTD_match_t* matches, U32 minml)
{
    (void)iLowLimit;
    switch(matchLengthSearch)
    {
    default :
    case 4 : return ZSTD_BtGetAllMatches_extDict(zc, ip, iHighLimit, maxNbAttempts, 4, matches, minml);
    case 5 : return ZSTD_BtGetAllMatches_extDict(zc, ip, iHighLimit, maxNbAttempts, 5, matches, minml);
    case 6 : return ZSTD_BtGetAllMatches_extDict(zc, ip, iHighLimit, maxNbAttempts, 6, matches, minml);
    }
}


/* ***********************
*  Hash Chain
*************************/
FORCE_INLINE /* inlining is important to hardwire a hot branch (template emulation) */
U32 ZSTD_HcGetAllMatches_generic (
                        ZSTD_CCtx* zc,   /* Index table will be updated */
                        const BYTE* const ip, const BYTE* const iLowLimit, const BYTE* const iHighLimit,
                        const U32 maxNbAttempts, const U32 mls, const U32 extDict, ZSTD_match_t* matches, size_t minml)
{
    U32* const chainTable = zc->contentTable;
    const U32 chainSize = (1U << zc->params.contentLog);
    const U32 chainMask = chainSize-1;
    const BYTE* const base = zc->base;
    const BYTE* const dictBase = zc->dictBase;
    const U32 dictLimit = zc->dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const dictStart  = dictBase + zc->lowLimit;
    const U32 lowLimit = zc->lowLimit;
    const U32 current = (U32)(ip-base);
    const U32 minChain = current > chainSize ? current - chainSize : 0;
    U32 matchIndex;
    U32 mnum = 0;
    const BYTE* match;
    U32 nbAttempts=maxNbAttempts;
    minml=MINMATCH-1;

    /* HC4 match finder */
    matchIndex = ZSTD_insertAndFindFirstIndex (zc, ip, mls);
    if (matchIndex >= current) return 0;

    while ((matchIndex>lowLimit) && (nbAttempts)) {
