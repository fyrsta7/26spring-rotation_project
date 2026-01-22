        return contribution;
#else
        return MAX(0, contribution); /* sometimes better, sometimes not ... */
#endif
    }
}

/* ZSTD_literalsContribution() :
 * creates a fake cost for the literals part of a sequence
 * which can be compared to the ending cost of a match
 * should a new match start at this position */
static int ZSTD_literalsContribution(const BYTE* const literals, U32 const litLength,
                                     const optState_t* const optPtr,
                                     int optLevel)
{
    int const contribution = ZSTD_rawLiteralsCost(literals, litLength, optPtr, optLevel)
                           + ZSTD_litLengthContribution(litLength, optPtr, optLevel);
    return contribution;
}

/* ZSTD_getMatchPrice() :
 * Provides the cost of the match part (offset + matchLength) of a sequence
 * Must be combined with ZSTD_fullLiteralsCost() to get the full cost of a sequence.
 * optLevel: when <2, favors small offset for decompression speed (improved cache efficiency) */
FORCE_INLINE_TEMPLATE U32
ZSTD_getMatchPrice(U32 const offset,
                   U32 const matchLength,
