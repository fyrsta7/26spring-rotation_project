
    return ip-istart;
}


typedef struct {
    size_t litLength;
    size_t matchLength;
    size_t offset;
} seq_t;

typedef struct {
    BIT_DStream_t DStream;
    FSE_DState_t stateLL;
    FSE_DState_t stateOffb;
    FSE_DState_t stateML;
    size_t prevOffset;
    const BYTE* dumps;
    const BYTE* dumpsEnd;
} seqState_t;


static void ZSTD_decodeSequence(seq_t* seq, seqState_t* seqState, const U32 mls)
{
    /* Literal length */
    U32 const litCode = FSE_peakSymbol(&(seqState->stateLL));
    {   static const U32 LL_base[MaxLL+1] = {
                             0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   10,    11,    12,    13,    14,     15,
                            16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000,
                            0x2000, 0x4000, 0x8000, 0x10000 };
        seq->litLength = LL_base[litCode] + BIT_readBits(&(seqState->DStream), LL_bits[litCode]);
    }

    /* Offset */
    {   static const U32 offsetPrefix[MaxOff+1] = {
                1 /*fake*/, 1, 2, 4, 8, 0x10, 0x20, 0x40,
                0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000,
                0x8000, 0x10000, 0x20000, 0x40000, 0x80000, 0x100000, 0x200000, 0x400000,
                0x800000, 0x1000000, 0x2000000, 0x4000000, /*fake*/ 1, 1, 1, 1 };
        U32 const offsetCode = FSE_peakSymbol(&(seqState->stateOffb));   /* <= maxOff, by table construction */
        U32 const nbBits = offsetCode ? offsetCode-1 : 0;
        size_t const offset = offsetCode ? offsetPrefix[offsetCode] + BIT_readBits(&(seqState->DStream), nbBits) :
                                           litCode ? seq->offset : seqState->prevOffset;
        if (offsetCode | !litCode) seqState->prevOffset = seq->offset;   /* cmove */
        seq->offset = offset;
        if (MEM_32bits()) BIT_reloadDStream(&(seqState->DStream));
        FSE_decodeSymbol(&(seqState->stateOffb), &(seqState->DStream));  /* update */
    }

    /* Literal length update */
    FSE_decodeSymbol(&(seqState->stateLL), &(seqState->DStream));   /* update */
    if (MEM_32bits()) BIT_reloadDStream(&(seqState->DStream));

    /* MatchLength */
    {   size_t matchLength = FSE_decodeSymbol(&(seqState->stateML), &(seqState->DStream));
        const BYTE* dumps = seqState->dumps;
        if (matchLength == MaxML) {
            const BYTE* const de = seqState->dumpsEnd;
            const U32 add = *dumps++;
            if (add < 255) matchLength += add;
            else {
                matchLength = MEM_readLE32(dumps) & 0xFFFFFF;  /* no pb : dumps is always followed by seq tables > 1 byte */
                if (matchLength&1) matchLength>>=1, dumps += 3;
                else matchLength = (U16)(matchLength)>>1, dumps += 2;
            }
            if (dumps >= de) dumps = de-1;   /* late correction, to avoid read overflow (data is now corrupted anyway) */
        }
        matchLength += mls;
        seq->matchLength = matchLength;
        seqState->dumps = dumps;
    }

#if 0   /* debug */
    {
        static U64 totalDecoded = 0;
        printf("pos %6u : %3u literals & match %3u bytes at distance %6u \n",
           (U32)(totalDecoded), (U32)litLength, (U32)matchLength, (U32)offset);
