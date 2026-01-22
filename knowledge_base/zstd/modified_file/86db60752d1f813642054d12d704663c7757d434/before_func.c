    ebs.estBlockSize += ebs.estLitSize + ZSTD_blockHeaderSize;
    return ebs;
}

static int ZSTD_needSequenceEntropyTables(ZSTD_fseCTablesMetadata_t const* fseMetadata)
{
    if (fseMetadata->llType == set_compressed || fseMetadata->llType == set_rle)
        return 1;
    if (fseMetadata->mlType == set_compressed || fseMetadata->mlType == set_rle)
        return 1;
    if (fseMetadata->ofType == set_compressed || fseMetadata->ofType == set_rle)
        return 1;
    return 0;
}

static size_t countLiterals(seqStore_t const* seqStore, const seqDef* sp, size_t seqCount)
{
    size_t n, total = 0;
    assert(sp != NULL);
    for (n=0; n<seqCount; n++) {
        total += ZSTD_getSequenceLength(seqStore, sp+n).litLength;
    }
    DEBUGLOG(6, "countLiterals for %zu sequences from %p => %zu bytes", seqCount, (const void*)sp, total);
    return total;
}

#define BYTESCALE 256

static size_t sizeBlockSequences(const seqDef* sp, size_t nbSeqs,
                size_t targetBudget, size_t avgLitCost, size_t avgSeqCost,
                int firstSubBlock)
{
    size_t n, budget = 0;
    /* entropy headers */
    if (firstSubBlock) {
        budget += 120 * BYTESCALE; /* generous estimate */
    }
    /* first sequence => at least one sequence*/
    budget += sp[0].litLength * avgLitCost + avgSeqCost;
    if (budget > targetBudget) return 1;

    /* loop over sequences */
    for (n=1; n<nbSeqs; n++) {
        size_t currentCost = sp[n].litLength * avgLitCost + avgSeqCost;
        if (budget + currentCost > targetBudget) break;
        budget += currentCost;
    }
    return n;
}

#define CBLOCK_TARGET_SIZE_MIN 1340 /* suitable to fit into an ethernet / wifi / 4G transport frame */

/** ZSTD_compressSubBlock_multi() :
 *  Breaks super-block into multiple sub-blocks and compresses them.
 *  Entropy will be written into the first block.
 *  The following blocks use repeat_mode to compress.
 *  Sub-blocks are all compressed, except the last one when beneficial.
 *  @return : compressed size of the super block (which features multiple ZSTD blocks)
 *            or 0 if it failed to compress. */
static size_t ZSTD_compressSubBlock_multi(const seqStore_t* seqStorePtr,
                            const ZSTD_compressedBlockState_t* prevCBlock,
                            ZSTD_compressedBlockState_t* nextCBlock,
                            const ZSTD_entropyCTablesMetadata_t* entropyMetadata,
                            const ZSTD_CCtx_params* cctxParams,
                                  void* dst, size_t dstCapacity,
                            const void* src, size_t srcSize,
                            const int bmi2, U32 lastBlock,
                            void* workspace, size_t wkspSize)
{
    const seqDef* const sstart = seqStorePtr->sequencesStart;
    const seqDef* const send = seqStorePtr->sequences;
    const seqDef* sp = sstart; /* tracks progresses within seqStorePtr->sequences */
    size_t const nbSeqs = (size_t)(send - sstart);
    const BYTE* const lstart = seqStorePtr->litStart;
    const BYTE* const lend = seqStorePtr->lit;
    const BYTE* lp = lstart;
    size_t const nbLiterals = (size_t)(lend - lstart);
    BYTE const* ip = (BYTE const*)src;
    BYTE const* const iend = ip + srcSize;
    BYTE* const ostart = (BYTE*)dst;
    BYTE* const oend = ostart + dstCapacity;
    BYTE* op = ostart;
    const BYTE* llCodePtr = seqStorePtr->llCode;
    const BYTE* mlCodePtr = seqStorePtr->mlCode;
    const BYTE* ofCodePtr = seqStorePtr->ofCode;
    size_t const minTarget = CBLOCK_TARGET_SIZE_MIN; /* enforce minimum size, to reduce undesirable side effects */
    size_t const targetCBlockSize = MAX(minTarget, cctxParams->targetCBlockSize);
    int writeLitEntropy = (entropyMetadata->hufMetadata.hType == set_compressed);
    int writeSeqEntropy = 1;

    DEBUGLOG(5, "ZSTD_compressSubBlock_multi (srcSize=%u, litSize=%u, nbSeq=%u)",
               (unsigned)srcSize, (unsigned)(lend-lstart), (unsigned)(send-sstart));

        /* let's start by a general estimation for the full block */
    if (nbSeqs > 0) {
        EstimatedBlockSize const ebs =
                ZSTD_estimateSubBlockSize(lp, nbLiterals,
                                        ofCodePtr, llCodePtr, mlCodePtr, nbSeqs,
                                        &nextCBlock->entropy, entropyMetadata,
                                        workspace, wkspSize,
                                        writeLitEntropy, writeSeqEntropy);
        /* quick estimation */
        size_t const avgLitCost = nbLiterals ? (ebs.estLitSize * BYTESCALE) / nbLiterals : BYTESCALE;
        size_t const avgSeqCost = ((ebs.estBlockSize - ebs.estLitSize) * BYTESCALE) / nbSeqs;
        const size_t nbSubBlocks = MAX((ebs.estBlockSize + (targetCBlockSize/2)) / targetCBlockSize, 1);
        size_t n, avgBlockBudget, blockBudgetSupp=0;
        avgBlockBudget = (ebs.estBlockSize * BYTESCALE) / nbSubBlocks;
        DEBUGLOG(5, "estimated fullblock size=%u bytes ; avgLitCost=%.2f ; avgSeqCost=%.2f ; targetCBlockSize=%u, nbSubBlocks=%u ; avgBlockBudget=%.0f bytes",
                    (unsigned)ebs.estBlockSize, (double)avgLitCost/BYTESCALE, (double)avgSeqCost/BYTESCALE,
                    (unsigned)targetCBlockSize, (unsigned)nbSubBlocks, (double)avgBlockBudget/BYTESCALE);

        /* compress and write sub-blocks */
        for (n=0; n+1 < nbSubBlocks; n++) {
            /* determine nb of sequences for current sub-block + nbLiterals from next sequence */
            size_t seqCount = sizeBlockSequences(sp, (size_t)(send-sp), avgBlockBudget + blockBudgetSupp, avgLitCost, avgSeqCost, n==0);
            /* if reached last sequence : break to last sub-block (simplification) */
            assert(seqCount <= (size_t)(send-sp));
            if (sp + seqCount == send) break;
            assert(seqCount > 0);
            /* compress sub-block */
            {   int litEntropyWritten = 0;
                int seqEntropyWritten = 0;
                size_t litSize = countLiterals(seqStorePtr, sp, seqCount);
                const size_t decompressedSize =
                        ZSTD_seqDecompressedSize(seqStorePtr, sp, seqCount, litSize, 0);
                size_t const cSize = ZSTD_compressSubBlock(&nextCBlock->entropy, entropyMetadata,
                                                sp, seqCount,
                                                lp, litSize,
                                                llCodePtr, mlCodePtr, ofCodePtr,
                                                cctxParams,
                                                op, (size_t)(oend-op),
                                                bmi2, writeLitEntropy, writeSeqEntropy,
                                                &litEntropyWritten, &seqEntropyWritten,
                                                0);
                FORWARD_IF_ERROR(cSize, "ZSTD_compressSubBlock failed");

                /* check compressibility, update state components */
                if (cSize > 0 && cSize < decompressedSize) {
                    DEBUGLOG(5, "Committed sub-block compressing %u bytes => %u bytes",
                                (unsigned)decompressedSize, (unsigned)cSize);
                    assert(ip + decompressedSize <= iend);
                    ip += decompressedSize;
                    lp += litSize;
                    op += cSize;
                    llCodePtr += seqCount;
                    mlCodePtr += seqCount;
                    ofCodePtr += seqCount;
                    /* Entropy only needs to be written once */
                    if (litEntropyWritten) {
                        writeLitEntropy = 0;
                    }
                    if (seqEntropyWritten) {
                        writeSeqEntropy = 0;
                    }
                    sp += seqCount;
                    blockBudgetSupp = 0;
            }   }
            /* otherwise : do not compress yet, coalesce current block with next one */
        }
    } /* if (nbSeqs > 0) */

    /* write last block */
    DEBUGLOG(2, "Generate last sub-block: %u sequences remaining", (unsigned)(send - sp));
    {   int litEntropyWritten = 0;
        int seqEntropyWritten = 0;
        size_t litSize = (size_t)(lend - lp);
        size_t seqCount = (size_t)(send - sp);
        const size_t decompressedSize =
                ZSTD_seqDecompressedSize(seqStorePtr, sp, seqCount, litSize, 1);
        size_t const cSize = ZSTD_compressSubBlock(&nextCBlock->entropy, entropyMetadata,
                                            sp, seqCount,
                                            lp, litSize,
                                            llCodePtr, mlCodePtr, ofCodePtr,
                                            cctxParams,
                                            op, (size_t)(oend-op),
                                            bmi2, writeLitEntropy, writeSeqEntropy,
                                            &litEntropyWritten, &seqEntropyWritten,
                                            lastBlock);
