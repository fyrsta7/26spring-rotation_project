    U32 const mlCode = FSE_peekSymbol(&(seqState->stateML));
    U32 const ofCode = FSE_peekSymbol(&(seqState->stateOffb));   /* <= maxOff, by table construction */

    U32 const llBits = LL_bits[llCode];
    U32 const mlBits = ML_bits[mlCode];
    U32 const ofBits = ofCode;
    U32 const totalBits = llBits+mlBits+ofBits;

    static const U32 LL_base[MaxLL+1] = {
                             0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   10,    11,    12,    13,    14,     15,
                            16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000,
                            0x2000, 0x4000, 0x8000, 0x10000 };

    static const U32 ML_base[MaxML+1] = {
                             3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,   14,    15,    16,    17,    18,
                            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,   30,    31,    32,    33,    34,
                            35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 0x83, 0x103, 0x203, 0x403, 0x803,
                            0x1003, 0x2003, 0x4003, 0x8003, 0x10003 };

    static const U32 OF_base[MaxOff+1] = {
                 0,        1,       1,       5,     0xD,     0x1D,     0x3D,     0x7D,
                 0xFD,   0x1FD,   0x3FD,   0x7FD,   0xFFD,   0x1FFD,   0x3FFD,   0x7FFD,
                 0xFFFD, 0x1FFFD, 0x3FFFD, 0x7FFFD, 0xFFFFD, 0x1FFFFD, 0x3FFFFD, 0x7FFFFD,
                 0xFFFFFD, 0x1FFFFFD, 0x3FFFFFD, 0x7FFFFFD, 0xFFFFFFD };

    /* sequence */
    {   size_t offset;
        if (!ofCode)
            offset = 0;
        else {
            offset = OF_base[ofCode] + BIT_readBits(&(seqState->DStream), ofBits);   /* <=  (ZSTD_WINDOWLOG_MAX-1) bits */
            if (MEM_32bits()) BIT_reloadDStream(&(seqState->DStream));
        }

        if (ofCode <= 1) {
            offset += (llCode==0);
            if (offset) {
                size_t const temp = (offset==3) ? seqState->prevOffset[0] - 1 : seqState->prevOffset[offset];
                if (offset != 1) seqState->prevOffset[2] = seqState->prevOffset[1];
                seqState->prevOffset[1] = seqState->prevOffset[0];
                seqState->prevOffset[0] = offset = temp;
            } else {
                offset = seqState->prevOffset[0];
            }
        } else {
            seqState->prevOffset[2] = seqState->prevOffset[1];
            seqState->prevOffset[1] = seqState->prevOffset[0];
            seqState->prevOffset[0] = offset;
        }
        seq.offset = offset;
    }

    seq.matchLength = ML_base[mlCode] + ((mlCode>31) ? BIT_readBits(&(seqState->DStream), mlBits) : 0);   /* <=  16 bits */
    if (MEM_32bits() && (mlBits+llBits>24)) BIT_reloadDStream(&(seqState->DStream));

    seq.litLength = LL_base[llCode] + ((llCode>15) ? BIT_readBits(&(seqState->DStream), llBits) : 0);   /* <=  16 bits */
    if (MEM_32bits() ||
       (totalBits > 64 - 7 - (LLFSELog+MLFSELog+OffFSELog)) ) BIT_reloadDStream(&(seqState->DStream));

    /* ANS state update */
    FSE_updateState(&(seqState->stateLL), &(seqState->DStream));   /* <=  9 bits */
    FSE_updateState(&(seqState->stateML), &(seqState->DStream));   /* <=  9 bits */
    if (MEM_32bits()) BIT_reloadDStream(&(seqState->DStream));     /* <= 18 bits */
    FSE_updateState(&(seqState->stateOffb), &(seqState->DStream)); /* <=  8 bits */

    return seq;
}


FORCE_INLINE
size_t ZSTD_execSequence(BYTE* op,
