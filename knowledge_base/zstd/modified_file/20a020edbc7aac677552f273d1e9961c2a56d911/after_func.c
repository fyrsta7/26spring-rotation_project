        for (hashIdx = 0; hashIdx < (1U << ms->cParams.hashLog); hashIdx += (1 << ZSTD_LAZY_DDSS_BUCKET_LOG)) {
            U32 count = 0;
            U32 i = hashTable[hashIdx + bucketSize - 1];
            while (i) {
                chainTable[chainPos++] = i;
                count++;
                if (i < minChain || count >= 255) {
                    break;
                }
                i = chains[i & chainMask];
            }
            hashTable[hashIdx + bucketSize - 1] = ((chainPos - count) << 8) + count;
        }
    }

    ms->nextToUpdate = target;
    free(chains);
}


/* inlining is important to hardwire a hot branch (template emulation) */
FORCE_INLINE_TEMPLATE
size_t ZSTD_HcFindBestMatch_generic (
                        ZSTD_matchState_t* ms,
                        const BYTE* const ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 mls, const ZSTD_dictMode_e dictMode)
{
    const ZSTD_compressionParameters* const cParams = &ms->cParams;
    U32* const chainTable = ms->chainTable;
    const U32 chainSize = (1 << cParams->chainLog);
    const U32 chainMask = chainSize-1;
    const BYTE* const base = ms->window.base;
    const BYTE* const dictBase = ms->window.dictBase;
    const U32 dictLimit = ms->window.dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const U32 curr = (U32)(ip-base);
    const U32 maxDistance = 1U << cParams->windowLog;
    const U32 lowestValid = ms->window.lowLimit;
    const U32 withinMaxDistance = (curr - lowestValid > maxDistance) ? curr - maxDistance : lowestValid;
    const U32 isDictionary = (ms->loadedDictEnd != 0);
    const U32 lowLimit = isDictionary ? lowestValid : withinMaxDistance;
    const U32 minChain = curr > chainSize ? curr - chainSize : 0;
    U32 nbAttempts = 1U << cParams->searchLog;
    size_t ml=4-1;

    const ZSTD_matchState_t* const dms = ms->dictMatchState;
    const U32 ddsHashLog = dictMode == ZSTD_dedicatedDictSearch
                         ? dms->cParams.hashLog - ZSTD_LAZY_DDSS_BUCKET_LOG : 0;
    const size_t ddsIdx = dictMode == ZSTD_dedicatedDictSearch
                        ? ZSTD_hashPtr(ip, ddsHashLog, mls) << ZSTD_LAZY_DDSS_BUCKET_LOG : 0;

    U32 matchIndex;

    if (dictMode == ZSTD_dedicatedDictSearch) {
        const U32* entry = &dms->hashTable[ddsIdx];
        PREFETCH_L1(entry);
    }

    /* HC4 match finder */
    matchIndex = ZSTD_insertAndFindFirstIndex_internal(ms, cParams, ip, mls);

    for ( ; (matchIndex>=lowLimit) & (nbAttempts>0) ; nbAttempts--) {
        size_t currentMl=0;
        if ((dictMode != ZSTD_extDict) || matchIndex >= dictLimit) {
            const BYTE* const match = base + matchIndex;
            assert(matchIndex >= dictLimit);   /* ensures this is true if dictMode != ZSTD_extDict */
            if (match[ml] == ip[ml])   /* potentially better */
                currentMl = ZSTD_count(ip, match, iLimit);
        } else {
            const BYTE* const match = dictBase + matchIndex;
            assert(match+4 <= dictEnd);
            if (MEM_read32(match) == MEM_read32(ip))   /* assumption : matchIndex <= dictLimit-4 (by table construction) */
                currentMl = ZSTD_count_2segments(ip+4, match+4, iLimit, dictEnd, prefixStart) + 4;
        }

        /* save best solution */
        if (currentMl > ml) {
            ml = currentMl;
            *offsetPtr = curr - matchIndex + ZSTD_REP_MOVE;
            if (ip+currentMl == iLimit) break; /* best possible, avoids read overflow on next attempt */
        }

        if (matchIndex <= minChain) break;
        matchIndex = NEXT_IN_CHAIN(matchIndex, chainMask);
    }

    if (dictMode == ZSTD_dedicatedDictSearch) {
        const U32 ddsLowestIndex  = dms->window.dictLimit;
        const BYTE* const ddsBase = dms->window.base;
        const BYTE* const ddsEnd  = dms->window.nextSrc;
        const U32 ddsSize         = (U32)(ddsEnd - ddsBase);
        const U32 ddsIndexDelta   = dictLimit - ddsSize;
        const U32 bucketSize      = (1 << ZSTD_LAZY_DDSS_BUCKET_LOG);
        const U32 bucketLimit     = nbAttempts < bucketSize - 1 ? nbAttempts : bucketSize - 1;
        U32 ddsAttempt;

        for (ddsAttempt = 0; ddsAttempt < bucketSize - 1; ddsAttempt++) {
            PREFETCH_L1(ddsBase + dms->hashTable[ddsIdx + ddsAttempt]);
        }

        {
            U32 const chainPackedPointer = dms->hashTable[ddsIdx + bucketSize - 1];
            U32 const chainIndex = chainPackedPointer >> 8;

            PREFETCH_L1(&dms->chainTable[chainIndex]);
        }

        for (ddsAttempt = 0; ddsAttempt < bucketLimit; ddsAttempt++) {
            size_t currentMl=0;
            const BYTE* match;
            matchIndex = dms->hashTable[ddsIdx + ddsAttempt];
            match = ddsBase + matchIndex;

            if (matchIndex < ddsLowestIndex) {
                return ml;
            }

            assert(match+4 <= ddsEnd);
            if (MEM_read32(match) == MEM_read32(ip)) {
                /* assumption : matchIndex <= dictLimit-4 (by table construction) */
                currentMl = ZSTD_count_2segments(ip+4, match+4, iLimit, ddsEnd, prefixStart) + 4;
            }

            /* save best solution */
            if (currentMl > ml) {
                ml = currentMl;
                *offsetPtr = curr - (matchIndex + ddsIndexDelta) + ZSTD_REP_MOVE;
                if (ip+currentMl == iLimit) {
                    /* best possible, avoids read overflow on next attempt */
                    return ml;
                }
            }
        }

        {
            U32 const chainPackedPointer = dms->hashTable[ddsIdx + bucketSize - 1];
            U32 chainIndex = chainPackedPointer >> 8;
            U32 const chainLength = chainPackedPointer & 0xFF;
            U32 const chainAttempts = nbAttempts - ddsAttempt;
            U32 const chainLimit = chainAttempts > chainLength ? chainLength : chainAttempts;
            U32 chainAttempt;

            for (chainAttempt = 0 ; chainAttempt < chainLimit; chainAttempt++) {
                PREFETCH_L1(ddsBase + dms->chainTable[chainIndex + chainAttempt]);
            }

            for (chainAttempt = 0 ; chainAttempt < chainLimit; chainAttempt++, chainIndex++) {
                size_t currentMl=0;
                const BYTE* match;
                matchIndex = dms->chainTable[chainIndex];
                match = ddsBase + matchIndex;

                if (matchIndex < ddsLowestIndex) {
                    break;
                }

                assert(match+4 <= ddsEnd);
                if (MEM_read32(match) == MEM_read32(ip)) {
                    /* assumption : matchIndex <= dictLimit-4 (by table construction) */
                    currentMl = ZSTD_count_2segments(ip+4, match+4, iLimit, ddsEnd, prefixStart) + 4;
                }

                /* save best solution */
                if (currentMl > ml) {
                    ml = currentMl;
                    *offsetPtr = curr - (matchIndex + ddsIndexDelta) + ZSTD_REP_MOVE;
                    if (ip+currentMl == iLimit) break; /* best possible, avoids read overflow on next attempt */
                }
            }
        }
    } else if (dictMode == ZSTD_dictMatchState) {
        const U32* const dmsChainTable = dms->chainTable;
        const U32 dmsChainSize         = (1 << dms->cParams.chainLog);
        const U32 dmsChainMask         = dmsChainSize - 1;
        const U32 dmsLowestIndex       = dms->window.dictLimit;
        const BYTE* const dmsBase      = dms->window.base;
        const BYTE* const dmsEnd       = dms->window.nextSrc;
        const U32 dmsSize              = (U32)(dmsEnd - dmsBase);
        const U32 dmsIndexDelta        = dictLimit - dmsSize;
        const U32 dmsMinChain = dmsSize > dmsChainSize ? dmsSize - dmsChainSize : 0;

        matchIndex = dms->hashTable[ZSTD_hashPtr(ip, dms->cParams.hashLog, mls)];

        for ( ; (matchIndex>=dmsLowestIndex) & (nbAttempts>0) ; nbAttempts--) {
