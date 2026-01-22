
#define FELEM int32_t
#define FELEM2 int64_t
#define FELEM_MAX INT32_MAX
#define FELEM_MIN INT32_MIN
#define WINDOW_TYPE 12
#endif


typedef struct AVResampleContext{
    FELEM *filter_bank;
