
  (void)strptime("1970-01-01 00:00:00", (const char *)("%Y-%m-%d %H:%M:%S"), &tm);
  m_deltaUtc = (int64_t)mktime(&tm);
  //printf("====delta:%lld\n\n", seconds);
  return;
}

static int64_t parseFraction(char* str, char** end, int32_t timePrec);
static int32_t parseTimeWithTz(char* timestr, int64_t* time, int32_t timePrec, char delim);
static int32_t parseLocaltime(char* timestr, int64_t* time, int32_t timePrec);
static int32_t parseLocaltimeWithDst(char* timestr, int64_t* time, int32_t timePrec);
static char* forwardToTimeStringEnd(char* str);
static bool checkTzPresent(char *str, int32_t len);

