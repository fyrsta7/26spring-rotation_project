        matchLength += ZSTD_count(ip+matchLength, iStart, iEnd);
    return matchLength;
}



/* *************************************
*  Hashes
***************************************/

static const U32 prime4bytes = 2654435761U;
static U32 ZSTD_hash4(U32 u, U32 h) { return (u * prime4bytes) >> (32-h) ; }
static size_t ZSTD_hash4Ptr(const void* ptr, U32 h) { return ZSTD_hash4(MEM_read32(ptr), h); }

static const U64 prime5bytes = 889523592379ULL;
static size_t ZSTD_hash5(U64 u, U32 h) { return (size_t)((u * prime5bytes) << (64-40) >> (64-h)) ; }
static size_t ZSTD_hash5Ptr(const void* p, U32 h) { return ZSTD_hash5(MEM_read64(p), h); }

static const U64 prime6bytes = 227718039650203ULL;
static size_t ZSTD_hash6(U64 u, U32 h) { return (size_t)((u * prime6bytes) << (64-48) >> (64-h)) ; }
static size_t ZSTD_hash6Ptr(const void* p, U32 h) { return ZSTD_hash6(MEM_read64(p), h); }

static const U64 prime7bytes = 58295818150454627ULL;
static size_t ZSTD_hash7(U64 u, U32 h) { return (size_t)((u * prime7bytes) << (64-56) >> (64-h)) ; }
static size_t ZSTD_hash7Ptr(const void* p, U32 h) { return ZSTD_hash7(MEM_read64(p), h); }

static size_t ZSTD_hashPtr(const void* p, U32 hBits, U32 mls)
{
    switch(mls)
    {
    default:
    case 4: return ZSTD_hash4Ptr(p, hBits);
    case 5: return ZSTD_hash5Ptr(p, hBits);
    case 6: return ZSTD_hash6Ptr(p, hBits);
    case 7: return ZSTD_hash7Ptr(p, hBits);
    }
}

/* *************************************
*  Fast Scan
***************************************/

#define FILLHASHSTEP 3
static void ZSTD_fillHashTable (ZSTD_CCtx* zc, const void* end, const U32 mls)
{
    U32* const hashTable = zc->hashTable;
    const U32 hBits = zc->params.hashLog;
    const BYTE* const base = zc->base;
    const BYTE* ip = base + zc->nextToUpdate;
    const BYTE* const iend = (const BYTE*) end;

    while(ip <= iend)
    {
        hashTable[ZSTD_hashPtr(ip, hBits, mls)] = (U32)(ip - base);
        ip += FILLHASHSTEP;
    }
}


FORCE_INLINE
size_t ZSTD_compressBlock_fast_generic(ZSTD_CCtx* zc,
                                       void* dst, size_t maxDstSize,
                                 const void* src, size_t srcSize,
                                 const U32 mls)
{
    U32* const hashTable = zc->hashTable;
    const U32 hBits = zc->params.hashLog;
    seqStore_t* seqStorePtr = &(zc->seqStore);
    const BYTE* const base = zc->base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32 lowIndex = zc->dictLimit;
    const BYTE* const lowest = base + lowIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;

    size_t offset_2=REPCODE_STARTVALUE, offset_1=REPCODE_STARTVALUE;


    /* init */
    ZSTD_resetSeqStore(seqStorePtr);
    if (ip < lowest+4)
    {
        hashTable[ZSTD_hashPtr(lowest+1, hBits, mls)] = lowIndex+1;
        hashTable[ZSTD_hashPtr(lowest+2, hBits, mls)] = lowIndex+2;
        hashTable[ZSTD_hashPtr(lowest+3, hBits, mls)] = lowIndex+3;
        ip = lowest+4;
    }

    /* Main Search Loop */
    while (ip < ilimit)  /* < instead of <=, because repcode check at (ip+1) */
    {
        size_t mlCode;
        size_t offset;
        const size_t h = ZSTD_hashPtr(ip, hBits, mls);
        const U32 matchIndex = hashTable[h];
        const BYTE* match = base + matchIndex;
        const U32 current = (U32)(ip-base);
        hashTable[h] = current;   /* update hash table */
