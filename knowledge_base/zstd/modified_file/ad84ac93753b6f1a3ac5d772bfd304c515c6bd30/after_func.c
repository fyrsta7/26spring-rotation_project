#endif


static int BMK_GetMilliSpan( int nTimeStart )
{
  int nSpan = BMK_GetMilliStart() - nTimeStart;
  if ( nSpan < 0 )
    nSpan += 0x100000 * 1000;
  return nSpan;
}

static U64 BMK_getFileSize(const char* infilename)
{
    int r;
#if defined(_MSC_VER)
    struct _stat64 statbuf;
    r = _stat64(infilename, &statbuf);
#else
    struct stat statbuf;
    r = stat(infilename, &statbuf);
#endif
    if (r || !S_ISREG(statbuf.st_mode)) return 0;   /* No good... */
    return (U64)statbuf.st_size;
}


/* ********************************************************
*  Bench functions
**********************************************************/
typedef struct
{
    const char* srcPtr;
    size_t srcSize;
    char*  cPtr;
    size_t cRoom;
    size_t cSize;
    char*  resPtr;
    size_t resSize;
} blockParam_t;

typedef size_t (*compressor_t) (void* dst, size_t maxDstSize, const void* src, size_t srcSize, int compressionLevel);

#define MIN(a,b) ((a)<(b) ? (a) : (b))

static int BMK_benchMem(const void* srcBuffer, size_t srcSize,
                        const char* displayName, int cLevel,
                        const size_t* fileSizes, U32 nbFiles)
{
    const size_t blockSize = (g_blockSize ? g_blockSize : srcSize) + (!srcSize);   /* avoid div by 0 */
    const U32 maxNbBlocks = (U32) ((srcSize + (blockSize-1)) / blockSize) + nbFiles;
    blockParam_t* const blockTable = (blockParam_t*) malloc(maxNbBlocks * sizeof(blockParam_t));
    const size_t maxCompressedSize = ZSTD_compressBound(srcSize) + (maxNbBlocks * 512);   /* add some room for safety */
    void* const compressedBuffer = malloc(maxCompressedSize);
    void* const resultBuffer = malloc(srcSize);
    const compressor_t compressor = ZSTD_compress;
    U64 crcOrig = XXH64(srcBuffer, srcSize, 0);
    U32 nbBlocks = 0;

    /* init */
    if (strlen(displayName)>17) displayName += strlen(displayName)-17;   /* can only display 17 characters */

    /* Memory allocation & restrictions */
    if (!compressedBuffer || !resultBuffer || !blockTable)
        EXM_THROW(31, "not enough memory");

    /* Init blockTable data */
    {
        U32 fileNb;
        const char* srcPtr = (const char*)srcBuffer;
        char* cPtr = (char*)compressedBuffer;
        char* resPtr = (char*)resultBuffer;
        for (fileNb=0; fileNb<nbFiles; fileNb++)
        {
            size_t remaining = fileSizes[fileNb];
            U32 nbBlocksforThisFile = (U32)((remaining + (blockSize-1)) / blockSize);
            U32 blockEnd = nbBlocks + nbBlocksforThisFile;
            for ( ; nbBlocks<blockEnd; nbBlocks++)
            {
                size_t thisBlockSize = MIN(remaining, blockSize);
                blockTable[nbBlocks].srcPtr = srcPtr;
                blockTable[nbBlocks].cPtr = cPtr;
                blockTable[nbBlocks].resPtr = resPtr;
                blockTable[nbBlocks].srcSize = thisBlockSize;
                blockTable[nbBlocks].cRoom = ZSTD_compressBound(thisBlockSize);
                srcPtr += thisBlockSize;
                cPtr += blockTable[nbBlocks].cRoom;
                resPtr += thisBlockSize;
                remaining -= thisBlockSize;
            }
        }
    }

    /* warmimg up memory */
    RDG_genBuffer(compressedBuffer, maxCompressedSize, 0.10, 0.50, 1);

    /* Bench */
    {
        int loopNb;
        size_t cSize = 0;
        double fastestC = 100000000., fastestD = 100000000.;
        double ratio = 0.;
        U64 crcCheck = 0;

        DISPLAY("\r%79s\r", "");
        for (loopNb = 1; loopNb <= nbIterations; loopNb++)
        {
            int nbLoops;
            int milliTime;
            U32 blockNb;

            /* Compression */
            DISPLAY("%2i-%-17.17s :%10u ->\r", loopNb, displayName, (U32)srcSize);
            memset(compressedBuffer, 0xE5, maxCompressedSize);

            nbLoops = 0;
            milliTime = BMK_GetMilliStart();
            while (BMK_GetMilliStart() == milliTime);
            milliTime = BMK_GetMilliStart();
            while (BMK_GetMilliSpan(milliTime) < TIMELOOP)
            {
                for (blockNb=0; blockNb<nbBlocks; blockNb++)
                    blockTable[blockNb].cSize = compressor(blockTable[blockNb].cPtr,  blockTable[blockNb].cRoom, blockTable[blockNb].srcPtr,blockTable[blockNb].srcSize, cLevel);
                nbLoops++;
            }
            milliTime = BMK_GetMilliSpan(milliTime);

            cSize = 0;
            for (blockNb=0; blockNb<nbBlocks; blockNb++)
                cSize += blockTable[blockNb].cSize;
            if ((double)milliTime < fastestC*nbLoops) fastestC = (double)milliTime / nbLoops;
            ratio = (double)srcSize / (double)cSize;
            DISPLAY("%2i-%-17.17s :%10i ->%10i (%5.3f),%6.1f MB/s\r", loopNb, displayName, (int)srcSize, (int)cSize, ratio, (double)srcSize / fastestC / 1000.);

#if 1
            /* Decompression */
            memset(resultBuffer, 0xD6, srcSize);

