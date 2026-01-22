    U32 back;
} ZSTD_match_t;

typedef struct {
    U32 price;
    U32 off;
    U32 mlen;
    U32 litlen;
    U32 rep;
    U32 rep2;
} ZSTD_optimal_t;


/*-  Constants  -*/
#define ZSTD_OPT_NUM   (1<<12)
#define ZSTD_FREQ_THRESHOLD (256)

/*-  Debug  -*/
#define ZSTD_OPT_DEBUG 0     // 1 = tableID=0;    5 = check encoded sequences
