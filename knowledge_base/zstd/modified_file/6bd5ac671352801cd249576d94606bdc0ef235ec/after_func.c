    int const result = !strncmp(*stringPtr, longCommand, comSize);
    if (result) *stringPtr += comSize;
    return result;
}


int usage(const char* exeName)
{
    DISPLAY (" \n");
    DISPLAY (" %s [Options] filename(s) \n", exeName);
    DISPLAY (" \n");
    DISPLAY ("Options : \n");
    DISPLAY ("-z          : benchmark compression (default) \n");
    DISPLAY ("-d          : benchmark decompression \n");
    DISPLAY ("-r          : recursively load all files in subdirectories (default: off) \n");
    DISPLAY ("-B#         : split input into blocks of size # (default: no split) \n");
    DISPLAY ("-#          : use compression level # (default: %u) \n", CLEVEL_DEFAULT);
    DISPLAY ("-D #        : use # as a dictionary (default: create one) \n");
    DISPLAY ("-i#         : nb benchmark rounds (default: %u) \n", BENCH_TIME_DEFAULT_S);
    DISPLAY ("--nbBlocks=#: use # blocks for bench (default: one per file) \n");
    DISPLAY ("--nbDicts=# : create # dictionaries for bench (default: one per block) \n");
    DISPLAY ("-h          : help (this text) \n");
    DISPLAY (" \n");
    DISPLAY ("Advanced Options (see zstd.h for documentation) : \n");
    DISPLAY ("--dedicated-dict-search\n");
    DISPLAY ("--dict-content-type=#\n");
    DISPLAY ("--dict-attach-pref=#\n");
    return 0;
}

int bad_usage(const char* exeName)
{
    DISPLAY (" bad usage : \n");
    usage(exeName);
    return 1;
}

int main (int argc, const char** argv)
{
    int recursiveMode = 0;
    int benchCompression = 1;
    int dedicatedDictSearch = 0;
    unsigned nbRounds = BENCH_TIME_DEFAULT_S;
    const char* const exeName = argv[0];

    if (argc < 2) return bad_usage(exeName);

    const char** nameTable = (const char**)malloc((size_t)argc * sizeof(const char*));
    assert(nameTable != NULL);
    unsigned nameIdx = 0;

    const char* dictionary = NULL;
    int cLevel = CLEVEL_DEFAULT;
    size_t blockSize = BLOCKSIZE_DEFAULT;
    unsigned nbDicts = 0;  /* determine nbDicts automatically: 1 dictionary per block */
    unsigned nbBlocks = 0; /* determine nbBlocks automatically, from source and blockSize */
    ZSTD_dictContentType_e dictContentType = ZSTD_dct_auto;
    ZSTD_dictAttachPref_e dictAttachPref = ZSTD_dictDefaultAttach;
    ZSTD_paramSwitch_e prefetchCDictTables = ZSTD_ps_auto;

    for (int argNb = 1; argNb < argc ; argNb++) {
        const char* argument = argv[argNb];
        if (!strcmp(argument, "-h")) { free(nameTable); return usage(exeName); }
        if (!strcmp(argument, "-d")) { benchCompression = 0; continue; }
        if (!strcmp(argument, "-z")) { benchCompression = 1; continue; }
        if (!strcmp(argument, "-r")) { recursiveMode = 1; continue; }
        if (!strcmp(argument, "-D")) { argNb++; assert(argNb < argc); dictionary = argv[argNb]; continue; }
        if (longCommandWArg(&argument, "-i")) { nbRounds = readU32FromChar(&argument); continue; }
        if (longCommandWArg(&argument, "--dictionary=")) { dictionary = argument; continue; }
        if (longCommandWArg(&argument, "-B")) { blockSize = readU32FromChar(&argument); continue; }
        if (longCommandWArg(&argument, "--blockSize=")) { blockSize = readU32FromChar(&argument); continue; }
        if (longCommandWArg(&argument, "--nbDicts=")) { nbDicts = readU32FromChar(&argument); continue; }
        if (longCommandWArg(&argument, "--nbBlocks=")) { nbBlocks = readU32FromChar(&argument); continue; }
