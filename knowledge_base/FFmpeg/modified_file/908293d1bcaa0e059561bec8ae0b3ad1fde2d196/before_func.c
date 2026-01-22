
typedef struct {
    char *line;         ///< null-terminated heap allocated subtitle line
    int64_t pos;        ///< offset position
    int start;          ///< timestamp start
    int end;            ///< timestamp end
} SubEntry;

typedef struct {
    int shift;
    unsigned timeres;
    SubEntry *subs;     ///< subtitles list
    int nsub;           ///< number of subtitles
    int sid;            ///< current subtitle
} JACOsubContext;

static int timed_line(const char *ptr)
