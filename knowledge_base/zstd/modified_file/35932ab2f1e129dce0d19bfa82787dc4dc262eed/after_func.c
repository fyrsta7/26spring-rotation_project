        ip1 = ip0 + 1;
    }

    /* save reps for next block */
    rep[0] = offset_1 ? offset_1 : offsetSaved;
    rep[1] = offset_2 ? offset_2 : offsetSaved;

    /* Return the last literals size */
    return (size_t)(iend - anchor);
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
 * Pos | Time -->
 * ----+-------------------
 * N   | ...4
 * N+1 | ... 3  4
 * N+2 | ...  2  3  4
 * N+3 |       1  2  3
 * N+4 |           1  2
 * N+5 |               1
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
 * N   | 1  2 3
 * N+1 |  1  2
 * N+2 |   1
 *
 * This is also the work we do at the beginning to enter the loop initially.
 */
FORCE_INLINE_TEMPLATE size_t
ZSTD_compressBlock_fast_generic_pipelined(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const mls)
{
    const ZSTD_compressionParameters* const cParams = &ms->cParams;
    U32* const hashTable = ms->hashTable;
    U32 const hlog = cParams->hashLog;
    /* support stepSize of 0 */
    size_t const stepSize = cParams->targetLength + !(cParams->targetLength);
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
    U32 current0;

    U32 rep_offset1 = rep[0];
    U32 rep_offset2 = rep[1];
    U32 offsetSaved = 0;

    size_t hash0; /* hash for ip0 */
    size_t hash1; /* hash for ip1 */
    size_t hash2; /* hash for ip2 */
    U32 idx0; /* match idx for ip0 */
    U32 idx1; /* match idx for ip1 */
    U32 mval; /* src value at match idx */

    U32 offcode;
    const BYTE* match0;
    size_t mLength;

    size_t step;
    const BYTE* nextStep;
    const size_t kStepIncr = (1 << (kSearchStrength - 1));

    DEBUGLOG(5, "ZSTD_compressBlock_fast_generic_pipelined");
    ip0 += (ip0 == prefixStart);
    {   U32 const curr = (U32)(ip0 - base);
        U32 const windowLow = ZSTD_getLowestPrefixIndex(ms, curr, cParams->windowLog);
        U32 const maxRep = curr - windowLow;
        if (rep_offset2 > maxRep) offsetSaved = rep_offset2, rep_offset2 = 0;
        if (rep_offset1 > maxRep) offsetSaved = rep_offset1, rep_offset1 = 0;
    }

    /* start each op */
_start: /* Requires: ip0 */

    step = stepSize;
    nextStep = ip0 + kStepIncr;

    /* calculate positions, ip0 - anchor == 0, so we skip step calc */
    ip1 = ip0 + stepSize;
    ip2 = ip1 + stepSize;

    if (ip2 >= ilimit) {
        goto _cleanup;
    }

    hash0 = ZSTD_hashPtr(ip0, hlog, mls);
    hash1 = ZSTD_hashPtr(ip1, hlog, mls);

    idx0 = hashTable[hash0];

    do {
        const U32 rval = MEM_read32(ip2 - rep_offset1);
        current0 = ip0 - base;

        /* write back hash table entry */
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
        if (idx0 >= prefixStartIndex) {
            mval = MEM_read32(base + idx0);
        } else {
            mval = MEM_read32(ip0) ^ 1; /* guaranteed to not match. */
        }

        /* check match at ip[0] */
        if (MEM_read32(ip0) == mval) {
            /* found a match! */
            goto _offset;
        }

        /* hash ip[2] */
        hash2 = ZSTD_hashPtr(ip2, hlog, mls);

        /* lookup ip[1] */
        idx1 = hashTable[hash1];

        /* advance to next positions */
        {
            if (ip1 >= nextStep) {
                PREFETCH_L1(ip1 + 64);
                step++;
                nextStep += kStepIncr;
            }

            idx0 = idx1;

            hash0 = hash1;
            hash1 = hash2;

            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip2 + step;
        }
    } while (ip2 < ilimit);

_cleanup:

    /* Find matches at end of block. */

    /* TODO */

    /* save reps for next block */
    rep[0] = rep_offset1 ? rep_offset1 : offsetSaved;
    rep[1] = rep_offset2 ? rep_offset2 : offsetSaved;

    /* Return the last literals size */
    return (size_t)(iend - anchor);
