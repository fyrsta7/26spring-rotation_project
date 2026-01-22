                ZSTD_memcpy(optPtr->offCodeFreq, baseOFCfreqs, sizeof(baseOFCfreqs));
                optPtr->offCodeSum = sum_u32(baseOFCfreqs, MaxOff+1);
            }


        }

    } else {   /* new block : re-use previous statistics, scaled down */

        if (compressedLiterals)
            optPtr->litSum = ZSTD_scaleStats(optPtr->litFreq, MaxLit, 12);
        optPtr->litLengthSum = ZSTD_scaleStats(optPtr->litLengthFreq, MaxLL, 11);
        optPtr->matchLengthSum = ZSTD_scaleStats(optPtr->matchLengthFreq, MaxML, 11);
        optPtr->offCodeSum = ZSTD_scaleStats(optPtr->offCodeFreq, MaxOff, 11);
    }

    ZSTD_setBasePrices(optPtr, optLevel);
}

/* ZSTD_rawLiteralsCost() :
 * price of literals (only) in specified segment (which length can be 0).
 * does not include price of literalLength symbol */
static U32 ZSTD_rawLiteralsCost(const BYTE* const literals, U32 const litLength,
                                const optState_t* const optPtr,
                                int optLevel)
