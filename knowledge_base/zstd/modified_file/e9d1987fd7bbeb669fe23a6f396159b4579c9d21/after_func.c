        }
    }
    return newReps;
}

/* update opt[pos] and last_pos */
#define SET_PRICE(pos, mlen_, offset_, litlen_, price_, rep_) \
    {                                                         \
        while (last_pos < pos)  { opt[last_pos+1].price = ZSTD_MAX_PRICE; last_pos++; } \
        opt[pos].mlen = mlen_;                                \
        opt[pos].off = offset_;                               \
        opt[pos].litlen = litlen_;                            \
        opt[pos].price = price_;                              \
        memcpy(opt[pos].rep, &rep_, sizeof(rep_));            \
    }

FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_opt_generic(ZSTD_CCtx* ctx,
                                      const void* src, size_t srcSize,
                                      const int optLevel, const int extDict)
{
    seqStore_t* const seqStorePtr = &(ctx->seqStore);
    optState_t* const optStatePtr = &(ctx->optState);
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ctx->base;
    const BYTE* const prefixStart = base + ctx->dictLimit;

    U32 const maxSearches = 1U << ctx->appliedParams.cParams.searchLog;
    U32 const sufficient_len = MIN(ctx->appliedParams.cParams.targetLength, ZSTD_OPT_NUM -1);
    U32 const mls = ctx->appliedParams.cParams.searchLength;
    U32 const minMatch = (ctx->appliedParams.cParams.searchLength == 3) ? 3 : 4;

    ZSTD_optimal_t* const opt = optStatePtr->priceTable;
    ZSTD_match_t* const matches = optStatePtr->matchTable;
    U32 rep[ZSTD_REP_NUM];

    /* init */
    DEBUGLOG(5, "ZSTD_compressBlock_opt_generic");
    ctx->nextToUpdate3 = ctx->nextToUpdate;
    ZSTD_rescaleFreqs(optStatePtr, (const BYTE*)src, srcSize);
    ip += (ip==prefixStart);
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) rep[i]=seqStorePtr->rep[i]; }

    /* Match Loop */
    while (ip < ilimit) {
        U32 cur, last_pos = 0;
        U32 best_mlen, best_off;

        /* find first match */
        {   U32 const litlen = (U32)(ip - anchor);
            U32 const ll0 = !litlen;
            U32 const nbMatches = ZSTD_BtGetAllMatches(ctx, ip, iend, extDict, maxSearches, mls, sufficient_len, rep, ll0, matches, minMatch);
            if (!nbMatches) { ip++; continue; }

            /* initialize opt[0] */
            { U32 i ; for (i=0; i<ZSTD_REP_NUM; i++) opt[0].rep[i] = rep[i]; }
            opt[0].mlen = 1;
            opt[0].litlen = litlen;

            /* large match -> immediate encoding */
            {   U32 const maxML = matches[nbMatches-1].len;
                DEBUGLOG(7, "found %u matches of maxLength=%u and offset=%u at cPos=%u => start new serie",
                            nbMatches, maxML, matches[nbMatches-1].off, (U32)(ip-prefixStart));

                if (maxML > sufficient_len) {
                    best_mlen = maxML;
                    best_off = matches[nbMatches-1].off;
                    DEBUGLOG(7, "large match (%u>%u), immediate encoding",
                                best_mlen, sufficient_len);
                    cur = 0;
                    last_pos = 1;
                    goto _shortestPath;
            }   }

            /* set prices for first matches starting position == 0 */
            {   U32 pos = minMatch;
                U32 matchNb;
                for (matchNb = 0; matchNb < nbMatches; matchNb++) {
                    U32 const offset = matches[matchNb].off;
                    U32 const end = matches[matchNb].len;
                    repcodes_t const repHistory = ZSTD_updateRep(rep, offset, ll0);
                    for ( ; pos <= end ; pos++) {
                        U32 const matchPrice = ZSTD_getPrice(optStatePtr, litlen, anchor, offset, pos, optLevel);
                        DEBUGLOG(7, "rPos:%u => set initial price : %u",
                                    pos, matchPrice);
                        SET_PRICE(pos, pos, offset, litlen, matchPrice, repHistory);   /* note : macro modifies last_pos */
            }   }   }
        }

        /* check further positions */
        for (cur = 1; cur <= last_pos; cur++) {
            const BYTE* const inr = ip + cur;
            assert(cur < ZSTD_OPT_NUM);

            /* Fix current position with one literal if cheaper */
            {   U32 const litlen = (opt[cur-1].mlen == 1) ? opt[cur-1].litlen + 1 : 1;
                U32 price;
                if (cur > litlen) {
                    price = opt[cur - litlen].price + ZSTD_getLiteralPrice(optStatePtr, litlen, inr-litlen);
                } else {
                    price = ZSTD_getLiteralPrice(optStatePtr, litlen, anchor);
                }
                if (price <= opt[cur].price) {
                    DEBUGLOG(7, "rPos:%u : better price (%u<%u) using literal",
                                cur, price, opt[cur].price);
                    SET_PRICE(cur, 1/*mlen*/, 0/*offset*/, litlen, price, opt[cur-1].rep);
            }   }

            /* last match must start at a minimum distance of 8 from oend */
            if (inr > ilimit) continue;

            if (cur == last_pos) break;

             if ( (optLevel==0) /*static*/
               && (opt[cur+1].price <= opt[cur].price) )
                continue;  /* skip unpromising positions; about ~+6% speed, -0.01 ratio */

            {   U32 const ll0 = (opt[cur].mlen != 1);
                U32 const litlen = (opt[cur].mlen == 1) ? opt[cur].litlen : 0;
                U32 const basePrice = (cur > litlen) ? opt[cur-litlen].price : 0;
                const BYTE* const baseLiterals = ip + cur - litlen;
                U32 const nbMatches = ZSTD_BtGetAllMatches(ctx, inr, iend, extDict, maxSearches, mls, sufficient_len, opt[cur].rep, ll0, matches, minMatch);
                U32 matchNb;
                if (!nbMatches) continue;
                assert(baseLiterals >= prefixStart);

                {   U32 const maxML = matches[nbMatches-1].len;
                    DEBUGLOG(7, "rPos:%u, found %u matches, of maxLength=%u",
                                cur, nbMatches, maxML);

                    if ( (maxML > sufficient_len)
                       | (cur + maxML >= ZSTD_OPT_NUM) ) {
                        best_mlen = maxML;
                        best_off = matches[nbMatches-1].off;
                        last_pos = cur + 1;
                        goto _shortestPath;
                    }
                }

                /* set prices using matches found at position == cur */
                for (matchNb = 0; matchNb < nbMatches; matchNb++) {
                    U32 const offset = matches[matchNb].off;
                    repcodes_t const repHistory = ZSTD_updateRep(opt[cur].rep, offset, ll0);
                    U32 const lastML = matches[matchNb].len;
                    U32 const startML = (matchNb>0) ? matches[matchNb-1].len+1 : minMatch;
                    U32 mlen;

                    DEBUGLOG(7, "testing match %u => offCode=%u, mlen=%u, llen=%u",
                                matchNb, matches[matchNb].off, lastML, litlen);

                    for (mlen = lastML; mlen >= startML; mlen--) {
                        U32 const pos = cur + mlen;
                        U32 const price = basePrice + ZSTD_getPrice(optStatePtr, litlen, baseLiterals, offset, mlen, optLevel);

                        if ((pos > last_pos) || (price < opt[pos].price)) {
                            DEBUGLOG(7, "rPos:%u => new better price (%u<%u)",
                                        pos, price, opt[pos].price);
                            SET_PRICE(pos, mlen, offset, litlen, price, repHistory);  /* note : macro modifies last_pos */
                        } else {
                            if (optLevel==0) break;  /* gets ~+10% speed for about -0.01 ratio loss */
                        }
            }   }   }
        }  /* for (cur = 1; cur <= last_pos; cur++) */

        best_mlen = opt[last_pos].mlen;
        best_off = opt[last_pos].off;
        cur = last_pos - best_mlen;

_shortestPath:   /* cur, last_pos, best_mlen, best_off have to be set */
        assert(opt[0].mlen == 1);

        /* reverse traversal */
        DEBUGLOG(7, "start reverse traversal (last_pos:%u, cur:%u)",
                    last_pos, cur);
        {   U32 selectedMatchLength = best_mlen;
            U32 selectedOffset = best_off;
            U32 pos = cur;
            while (1) {
                U32 const mlen = opt[pos].mlen;
                U32 const off = opt[pos].off;
                opt[pos].mlen = selectedMatchLength;
                opt[pos].off = selectedOffset;
                selectedMatchLength = mlen;
                selectedOffset = off;
                if (mlen > pos) break;
                pos -= mlen;
        }   }

        /* save sequences */
        {   U32 pos;
            for (pos=0; pos < last_pos; ) {
                U32 const llen = (U32)(ip - anchor);
                U32 const mlen = opt[pos].mlen;
                U32 const offset = opt[pos].off;
                if (mlen == 1) { ip++; pos++; continue; }  /* literal position => move on */
                pos += mlen; ip += mlen;

                /* repcodes update : like ZSTD_updateRep(), but update in place */
                if (offset >= ZSTD_REP_NUM) {  /* full offset */
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = offset - ZSTD_REP_MOVE;
                } else {   /* repcode */
                    U32 const repCode = offset + (llen==0);
                    if (repCode) {  /* note : if repCode==0, no change */
                        U32 const currentOffset = (repCode==ZSTD_REP_NUM) ? (rep[0] - 1) : rep[repCode];
                        if (repCode >= 2) rep[2] = rep[1];
                        rep[1] = rep[0];
                        rep[0] = currentOffset;
