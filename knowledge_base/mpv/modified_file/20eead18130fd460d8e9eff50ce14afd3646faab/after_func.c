#endif

struct priv {
    int fd;
    bool close;
    bool use_poll;
    bool regular_file;
    bool appending;
    int64_t cached_size; // -2: invalid, -1: unknown
    int64_t orig_size;
    struct mp_cancel *cancel;
};

// Total timeout = RETRY_TIMEOUT * MAX_RETRIES
#define RETRY_TIMEOUT 0.2
