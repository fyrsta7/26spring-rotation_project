        match0--;
        mLength++;
    }

_match: /* Requires: ip0, match0, offcode */

    /* Count the forward length. */
    mLength += ZSTD_count(ip0 + mLength, match0 + mLength, iend);

    ZSTD_storeSeq(seqStore, (size_t)(ip0 - anchor), anchor, iend, offcode, mLength);

    ip0 += mLength;
    anchor = ip0;

    /* Fill table and check for immediate repcode. */
    if (ip0 <= ilimit) {
        /* Fill Table */
        assert(base+current0+2 > istart);  /* check base overflow */
        hashTable[ZSTD_hashPtr(base+current0+2, hlog, mls)] = current0+2;  /* here because current+2 could be > iend-8 */
        hashTable[ZSTD_hashPtr(ip0-2, hlog, mls)] = (U32)(ip0-2-base);

        if (rep_offset2 > 0) { /* rep_offset2==0 means rep_offset2 is invalidated */
            while ( (ip0 <= ilimit) && (MEM_read32(ip0) == MEM_read32(ip0 - rep_offset2)) ) {
                /* store sequence */
                size_t const rLength = ZSTD_count(ip0+4, ip0+4-rep_offset2, iend) + 4;
                { U32 const tmpOff = rep_offset2; rep_offset2 = rep_offset1; rep_offset1 = tmpOff; } /* swap rep_offset2 <=> rep_offset1 */
                hashTable[ZSTD_hashPtr(ip0, hlog, mls)] = (U32)(ip0-base);
                ip0 += rLength;
                ZSTD_storeSeq(seqStore, 0 /*litLen*/, anchor, iend, REPCODE1_TO_OFFBASE, rLength);
                anchor = ip0;
                continue;   /* faster when present (confirmed on gcc-8) ... (?) */
    }   }   }

    goto _start;
}

#define ZSTD_GEN_FAST_FN(dictMode, mml, cmov)                                                       \
    static size_t ZSTD_compressBlock_fast_##dictMode##_##mml##_##cmov(                              \
            ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],                    \
            void const* src, size_t srcSize)                                                       \
    {                                                                                              \
        return ZSTD_compressBlock_fast_##dictMode##_generic(ms, seqStore, rep, src, srcSize, mml, cmov); \
    }

ZSTD_GEN_FAST_FN(noDict, 4, 1)
ZSTD_GEN_FAST_FN(noDict, 5, 1)
ZSTD_GEN_FAST_FN(noDict, 6, 1)
ZSTD_GEN_FAST_FN(noDict, 7, 1)

ZSTD_GEN_FAST_FN(noDict, 4, 0)
ZSTD_GEN_FAST_FN(noDict, 5, 0)
ZSTD_GEN_FAST_FN(noDict, 6, 0)
ZSTD_GEN_FAST_FN(noDict, 7, 0)

size_t ZSTD_compressBlock_fast(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    U32 const mml = ms->cParams.minMatch;
    /* use cmov when "candidate in range" branch is likely unpredictable */
    int const useCmov = ms->cParams.windowLog < 19;
    assert(ms->dictMatchState == NULL);
    if (useCmov) {
        switch(mml)
        {
        default: /* includes case 3 */
        case 4 :
            return ZSTD_compressBlock_fast_noDict_4_1(ms, seqStore, rep, src, srcSize);
        case 5 :
            return ZSTD_compressBlock_fast_noDict_5_1(ms, seqStore, rep, src, srcSize);
        case 6 :
            return ZSTD_compressBlock_fast_noDict_6_1(ms, seqStore, rep, src, srcSize);
        case 7 :
            return ZSTD_compressBlock_fast_noDict_7_1(ms, seqStore, rep, src, srcSize);
        }
    } else {
        /* use a branch instead */
        switch(mml)
        {
        default: /* includes case 3 */
        case 4 :
            return ZSTD_compressBlock_fast_noDict_4_0(ms, seqStore, rep, src, srcSize);
        case 5 :
            return ZSTD_compressBlock_fast_noDict_5_0(ms, seqStore, rep, src, srcSize);
        case 6 :
            return ZSTD_compressBlock_fast_noDict_6_0(ms, seqStore, rep, src, srcSize);
        case 7 :
            return ZSTD_compressBlock_fast_noDict_7_0(ms, seqStore, rep, src, srcSize);
        }
    }
}

FORCE_INLINE_TEMPLATE
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
size_t ZSTD_compressBlock_fast_dictMatchState_generic(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize, U32 const mls, U32 const hasStep)
{
    const ZSTD_compressionParameters* const cParams = &ms->cParams;
    U32* const hashTable = ms->hashTable;
    U32 const hlog = cParams->hashLog;
    /* support stepSize of 0 */
    U32 const stepSize = cParams->targetLength + !(cParams->targetLength);
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip0 = istart;
    const BYTE* ip1 = ip0 + stepSize; /* we assert below that stepSize >= 1 */
    const BYTE* anchor = istart;
    const U32   prefixStartIndex = ms->window.dictLimit;
    const BYTE* const prefixStart = base + prefixStartIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=rep[0], offset_2=rep[1];

    const ZSTD_matchState_t* const dms = ms->dictMatchState;
    const ZSTD_compressionParameters* const dictCParams = &dms->cParams ;
    const U32* const dictHashTable = dms->hashTable;
    const U32 dictStartIndex       = dms->window.dictLimit;
    const BYTE* const dictBase     = dms->window.base;
    const BYTE* const dictStart    = dictBase + dictStartIndex;
    const BYTE* const dictEnd      = dms->window.nextSrc;
    const U32 dictIndexDelta       = prefixStartIndex - (U32)(dictEnd - dictBase);
    const U32 dictAndPrefixLength  = (U32)(istart - prefixStart + dictEnd - dictStart);
    const U32 dictHBits            = dictCParams->hashLog + ZSTD_SHORT_CACHE_TAG_BITS;

    /* if a dictionary is still attached, it necessarily means that
     * it is within window size. So we just check it. */
    const U32 maxDistance = 1U << cParams->windowLog;
    const U32 endIndex = (U32)((size_t)(istart - base) + srcSize);
    assert(endIndex - prefixStartIndex <= maxDistance);
    (void)maxDistance; (void)endIndex;   /* these variables are not used when assert() is disabled */

    (void)hasStep; /* not currently specialized on whether it's accelerated */

    /* ensure there will be no underflow
     * when translating a dict index into a local index */
    assert(prefixStartIndex >= (U32)(dictEnd - dictBase));

    if (ms->prefetchCDictTables) {
        size_t const hashTableBytes = (((size_t)1) << dictCParams->hashLog) * sizeof(U32);
        PREFETCH_AREA(dictHashTable, hashTableBytes);
    }

    /* init */
    DEBUGLOG(5, "ZSTD_compressBlock_fast_dictMatchState_generic");
    ip0 += (dictAndPrefixLength == 0);
    /* dictMatchState repCode checks don't currently handle repCode == 0
     * disabling. */
    assert(offset_1 <= dictAndPrefixLength);
    assert(offset_2 <= dictAndPrefixLength);

    /* Outer search loop */
    assert(stepSize >= 1);
    while (ip1 <= ilimit) {   /* repcode check at (ip0 + 1) is safe because ip0 < ip1 */
        size_t mLength;
        size_t hash0 = ZSTD_hashPtr(ip0, hlog, mls);

        size_t const dictHashAndTag0 = ZSTD_hashPtr(ip0, dictHBits, mls);
        U32 dictMatchIndexAndTag = dictHashTable[dictHashAndTag0 >> ZSTD_SHORT_CACHE_TAG_BITS];
        int dictTagsMatch = ZSTD_comparePackedTags(dictMatchIndexAndTag, dictHashAndTag0);

        U32 matchIndex = hashTable[hash0];
        U32 curr = (U32)(ip0 - base);
        size_t step = stepSize;
        const size_t kStepIncr = 1 << kSearchStrength;
        const BYTE* nextStep = ip0 + kStepIncr;

        /* Inner search loop */
        while (1) {
            const BYTE* match = base + matchIndex;
            const U32 repIndex = curr + 1 - offset_1;
            const BYTE* repMatch = (repIndex < prefixStartIndex) ?
                                   dictBase + (repIndex - dictIndexDelta) :
                                   base + repIndex;
            const size_t hash1 = ZSTD_hashPtr(ip1, hlog, mls);
            size_t const dictHashAndTag1 = ZSTD_hashPtr(ip1, dictHBits, mls);
            hashTable[hash0] = curr;   /* update hash table */

            if ((ZSTD_index_overlap_check(prefixStartIndex, repIndex))
                && (MEM_read32(repMatch) == MEM_read32(ip0 + 1))) {
                const BYTE* const repMatchEnd = repIndex < prefixStartIndex ? dictEnd : iend;
                mLength = ZSTD_count_2segments(ip0 + 1 + 4, repMatch + 4, iend, repMatchEnd, prefixStart) + 4;
                ip0++;
                ZSTD_storeSeq(seqStore, (size_t) (ip0 - anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
                break;
            }

            if (dictTagsMatch) {
                /* Found a possible dict match */
                const U32 dictMatchIndex = dictMatchIndexAndTag >> ZSTD_SHORT_CACHE_TAG_BITS;
                const BYTE* dictMatch = dictBase + dictMatchIndex;
                if (dictMatchIndex > dictStartIndex &&
                    MEM_read32(dictMatch) == MEM_read32(ip0)) {
                    /* To replicate extDict parse behavior, we only use dict matches when the normal matchIndex is invalid */
                    if (matchIndex <= prefixStartIndex) {
