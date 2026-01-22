#define FELEM double
#define FELEM2 double
#define FELEML double
#define WINDOW_TYPE 24
#endif


typedef struct AVResampleContext{
    const AVClass *av_class;
    FELEM *filter_bank;
    int filter_length;
    int ideal_dst_incr;
    int dst_incr;
    int index;
