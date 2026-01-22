            if (i == 0)
                hashSmall[smHash] = curr + i;
            if (i == 0 || hashLarge[lgHash] == 0)
                hashLarge[lgHash] = curr + i;
            /* Only load extra positions for ZSTD_dtlm_full */
            if (dtlm == ZSTD_dtlm_fast)
                break;
    }   }
}


FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_doubleFast_singleSegment_generic(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const mls /* template */)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = ms->hashTable;
    const U32 hBitsL = cParams->hashLog;
    U32* const hashSmall = ms->chainTable;
    const U32 hBitsS = cParams->chainLog;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* anchor = istart;
    const U32 endIndex = (U32)((size_t)(istart - base) + srcSize);
    /* presumes that, if there is a dictionary, it must be using Attach mode */
    const U32 prefixLowestIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams->windowLog);
    const BYTE* const prefixLowest = base + prefixLowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=rep[0], offset_2=rep[1];
    U32 offsetSaved = 0;

    size_t mLength;
    U32 offset;
    U32 curr;

    const size_t kStepIncr = 1 << kSearchStrength;
    const BYTE* nextStep;
    size_t step;

    size_t hl0;
    size_t hs0;
    size_t hl1;
    // size_t hs1;

    U32 idxl0;
    U32 idxs0;
    U32 idxl1;
    // U32 idxs0;

    const BYTE* matchl0;
    const BYTE* matchs0;
    const BYTE* matchl1;
    // const BYTE* matchs1;

    const BYTE* ip = istart;
    const BYTE* ip1;

    DEBUGLOG(5, "ZSTD_compressBlock_doubleFast_singleSegment_generic");

    /* init */
    ip += ((ip - prefixLowest) == 0);
    {
        U32 const current = (U32)(ip - base);
        U32 const windowLow = ZSTD_getLowestPrefixIndex(ms, current, cParams->windowLog);
        U32 const maxRep = current - windowLow;
        if (offset_2 > maxRep) offsetSaved = offset_2, offset_2 = 0;
        if (offset_1 > maxRep) offsetSaved = offset_1, offset_1 = 0;
    }

_start:

    step = 1;
    nextStep = ip + kStepIncr;
    ip1 = ip + step;

    if (ip1 >= ilimit) {
        goto _cleanup;
    }

    hl0 = ZSTD_hashPtr(ip, hBitsL, 8);

    /* Main Search Loop */
    do {
        curr = (U32)(ip-base);
        hs0 = ZSTD_hashPtr(ip, hBitsS, mls);
        idxl0 = hashLong[hl0];
        idxs0 = hashSmall[hs0];
        matchl0 = base + idxl0;
        matchs0 = base + idxs0;

        hashLong[hl0] = hashSmall[hs0] = curr;   /* update hash tables */

        /* check noDict repcode */
        if ((offset_1 > 0) & (MEM_read32(ip+1-offset_1) == MEM_read32(ip+1))) {
            mLength = ZSTD_count(ip+1+4, ip+1+4-offset_1, iend) + 4;
            ip++;
            ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, 0, mLength-MINMATCH);
            goto _match_stored;
        }

        hl1 = ZSTD_hashPtr(ip1, hBitsL, 8);

        if (idxl0 > prefixLowestIndex) {
            /* check prefix long match */
            if (MEM_read64(matchl0) == MEM_read64(ip)) {
                mLength = ZSTD_count(ip+8, matchl0+8, iend) + 8;
                offset = (U32)(ip-matchl0);
                while (((ip>anchor) & (matchl0>prefixLowest)) && (ip[-1] == matchl0[-1])) { ip--; matchl0--; mLength++; } /* catch up */
                goto _match_found;
            }
        }

        if (idxs0 > prefixLowestIndex) {
            /* check prefix short match */
            if (MEM_read32(matchs0) == MEM_read32(ip)) {
                goto _search_next_long;
            }
        }

        if (ip1 >= nextStep) {
            PREFETCH_L1(ip1 + 64);
            PREFETCH_L1(ip1 + 128);
            step++;
            nextStep += kStepIncr;
        }
        ip = ip1;
        ip1 += step;

        hl0 = hl1;
#if defined(__aarch64__)
        PREFETCH_L1(ip+256);
#endif
    } while (ip1 < ilimit);

_cleanup:
    /* save reps for next block */
    rep[0] = offset_1 ? offset_1 : offsetSaved;
    rep[1] = offset_2 ? offset_2 : offsetSaved;

    /* Return the last literals size */
    return (size_t)(iend - anchor);

_search_next_long:
    {   idxl1 = hashLong[hl1];
        matchl1 = base + idxl1;

        /* check prefix long +1 match */
        if (idxl1 > prefixLowestIndex) {
            if (MEM_read64(matchl1) == MEM_read64(ip1)) {
                ip = ip1;
                mLength = ZSTD_count(ip+8, matchl1+8, iend) + 8;
                offset = (U32)(ip-matchl1);
                while (((ip>anchor) & (matchl1>prefixLowest)) && (ip[-1] == matchl1[-1])) { ip--; matchl1--; mLength++; } /* catch up */
                goto _match_found;
            }
        }
    }

    /* if no long +1 match, explore the short match we found */
    {
        mLength = ZSTD_count(ip+4, matchs0+4, iend) + 4;
        offset = (U32)(ip - matchs0);
        while (((ip>anchor) & (matchs0>prefixLowest)) && (ip[-1] == matchs0[-1])) { ip--; matchs0--; mLength++; } /* catch up */
    }

    /* fall-through */

_match_found: /* requires ip, offset, mLength */
    offset_2 = offset_1;
    offset_1 = offset;

    if (step < 4) {
        /* It is unsafe to write this value back to the hashtable when ip1 is
         * greater than or equal to the new ip we will have after we're done
         * processing this match. Rather than perform that test directly
         * (ip1 >= ip + mLength), which costs speed in practice, we do a simpler
         * more predictable test. The minmatch even if we take a short match is
         * 4 bytes, so as long as step, the distance between ip and ip1
         * (initially) is less than 4, we know ip1 < new ip. */
        hashLong[hl1] = (U32)(ip1 - base);
    }

    ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, offset + ZSTD_REP_MOVE, mLength-MINMATCH);

_match_stored:
    /* match found */
    ip += mLength;
    anchor = ip;

    if (ip <= ilimit) {
        /* Complementary insertion */
        /* done after iLimit test, as candidates could be > iend-8 */
        {   U32 const indexToInsert = curr+2;
            hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
            hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
            hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
            hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
        }

