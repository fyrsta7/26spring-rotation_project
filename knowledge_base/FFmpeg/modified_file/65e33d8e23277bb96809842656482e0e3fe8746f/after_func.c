#    define DELEM  int32_t
#    define FELEM  int32_t
#    define FELEM2 int64_t
#    define FELEM_MAX INT32_MAX
#    define FELEM_MIN INT32_MIN
#    define FOFFSET (1<<(FILTER_SHIFT-1))
#    define OUT(d, v) (d) = av_clipl_int32((v)>>FILTER_SHIFT)

#elif    defined(TEMPLATE_RESAMPLE_S16)

#    define RENAME(N) N ## _int16
#    define FILTER_SHIFT 15
