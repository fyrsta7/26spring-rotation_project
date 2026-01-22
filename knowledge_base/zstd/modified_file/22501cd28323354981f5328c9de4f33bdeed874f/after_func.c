        {   U32 p;
            for (p = 1; p < fastHashFillStep; ++p) {
                size_t const hash = ZSTD_hashPtr(ip + p, hBits, mls);
                if (hashTable[hash] == 0) {  /* not yet filled */
                    hashTable[hash] = curr + p;
    }   }   }   }
}


/**
 * If you squint hard enough (and ignore repcodes), the search operation at any
 * given position is broken into 4 stages:
 *
 * 1. Hash   (map position to hash value via input read)
 * 2. Lookup (map hash val to index via hashtable read)
 * 3. Load   (map index to value at that position via input read)
 * 4. Compare
 *
 * Each of these steps involves a memory read at an address which is computed
 * from the previous step. This means these steps must be sequenced and their
 * latencies are cumulative.
 *
 * Rather than do 1->2->3->4 sequentially for a single position before moving
 * onto the next, this implementation interleaves these operations across the
 * next few positions:
 *
 * R = Repcode Read & Compare
 * H = Hash
 * T = Table Lookup
 * M = Match Read & Compare
 *
 * Pos | Time -->
 * ----+-------------------
 * N   | ... M
 * N+1 | ...   TM
 * N+2 |    R H   T M
 * N+3 |         H    TM
 * N+4 |           R H   T M
 * N+5 |                H   ...
 * N+6 |                  R ...
 *
 * This is very much analogous to the pipelining of execution in a CPU. And just
 * like a CPU, we have to dump the pipeline when we find a match (i.e., take a
 * branch).
 *
 * When this happens, we throw away our current state, and do the following prep
 * to re-enter the loop:
 *
 * Pos | Time -->
 * ----+-------------------
 * N   | H T
 * N+1 |  H
 *
 * This is also the work we do at the beginning to enter the loop initially.
 */
FORCE_INLINE_TEMPLATE size_t
ZSTD_compressBlock_fast_noDict_generic(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const mls)
{
    const ZSTD_compressionParameters* const cParams = &ms->cParams;
    U32* const hashTable = ms->hashTable;
    U32 const hlog = cParams->hashLog;
    /* support stepSize of 0 */
    size_t const stepSize = cParams->targetLength + !(cParams->targetLength) - 1;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const U32   endIndex = (U32)((size_t)(istart - base) + srcSize);
    const U32   prefixStartIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams->windowLog);
    const BYTE* const prefixStart = base + prefixStartIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;

    const BYTE* anchor = istart;
    const BYTE* ip0 = istart;
    const BYTE* ip1;
    const BYTE* ip2;
    const BYTE* ip3;
    U32 current0;

    U32 rep_offset1 = rep[0];
    U32 rep_offset2 = rep[1];
    U32 offsetSaved = 0;

    size_t hash0; /* hash for ip0 */
    size_t hash1; /* hash for ip1 */
    U32 idx; /* match idx for ip0 */
    U32 mval; /* src value at match idx */

    U32 offcode;
    const BYTE* match0;
    size_t mLength;

    size_t step;
    const BYTE* nextStep;
    const size_t kStepIncr = (1 << (kSearchStrength - 1));

    DEBUGLOG(5, "ZSTD_compressBlock_fast_generic");
    ip0 += (ip0 == prefixStart);
    {   U32 const curr = (U32)(ip0 - base);
        U32 const windowLow = ZSTD_getLowestPrefixIndex(ms, curr, cParams->windowLog);
        U32 const maxRep = curr - windowLow;
        if (rep_offset2 > maxRep) offsetSaved = rep_offset2, rep_offset2 = 0;
        if (rep_offset1 > maxRep) offsetSaved = rep_offset1, rep_offset1 = 0;
    }

    /* start each op */
_start: /* Requires: ip0 */

    step = 1;
    nextStep = ip0 + kStepIncr;

    /* calculate positions, ip0 - anchor == 0, so we skip step calc */
    ip1 = ip0 + step;
    ip2 = ip1 + step + stepSize;
    ip3 = ip2 + step;

    if (ip3 >= ilimit) {
        goto _cleanup;
    }

    hash0 = ZSTD_hashPtr(ip0, hlog, mls);
    hash1 = ZSTD_hashPtr(ip1, hlog, mls);

    idx = hashTable[hash0];

    do {
        /* load repcode match for ip[2]*/
        const U32 rval = MEM_read32(ip2 - rep_offset1);

        /* write back hash table entry */
        current0 = (U32)(ip0 - base);
        hashTable[hash0] = current0;

        /* check repcode at ip[2] */
        if ((MEM_read32(ip2) == rval) & (rep_offset1 > 0)) {
            ip0 = ip2;
            match0 = ip0 - rep_offset1;
            mLength = ip0[-1] == match0[-1];
            ip0 -= mLength;
            match0 -= mLength;
            offcode = 0;
            mLength += 4;
            goto _match;
        }

        /* load match for ip[0] */
        if (idx >= prefixStartIndex) {
            mval = MEM_read32(base + idx);
        } else {
            mval = MEM_read32(ip0) ^ 1; /* guaranteed to not match. */
        }

        /* check match at ip[0] */
        if (MEM_read32(ip0) == mval) {
            /* found a match! */
            goto _offset;
        }

        /* lookup ip[1] */
        idx = hashTable[hash1];

        /* hash ip[2] */
        hash0 = hash1;
        hash1 = ZSTD_hashPtr(ip2, hlog, mls);

        /* advance to next positions */
        ip0 = ip1;
        ip1 = ip2;
        ip2 = ip3;

        /* write back hash table entry */
        current0 = (U32)(ip0 - base);
        hashTable[hash0] = current0;

        /* load match for ip[0] */
        if (idx >= prefixStartIndex) {
            mval = MEM_read32(base + idx);
        } else {
            mval = MEM_read32(ip0) ^ 1; /* guaranteed to not match. */
        }

        /* check match at ip[0] */
        if (MEM_read32(ip0) == mval) {
            /* found a match! */
            goto _offset;
        }

        /* lookup ip[1] */
        idx = hashTable[hash1];

        /* hash ip[2] */
        hash0 = hash1;
        hash1 = ZSTD_hashPtr(ip2, hlog, mls);

        /* calculate step */
        if (ip2 >= nextStep) {
            PREFETCH_L1(ip1 + 64);
            PREFETCH_L1(ip1 + 128);
            step++;
            nextStep += kStepIncr;
        }

        /* advance to next positions */
        ip0 = ip1;
        ip1 = ip2;
        ip2 = ip2 + step + stepSize;
        ip3 = ip2 + step;
    } while (ip3 < ilimit);

_cleanup:
    /* Note that there are probably still a couple positions we could search.
     * However, it seems to be a meaningful performance hit to try to search
     * them. So let's not. */

    /* save reps for next block */
    rep[0] = rep_offset1 ? rep_offset1 : offsetSaved;
