        OPT_STRING("tls-key-file", tls_key_file, M_OPT_FILE),
        OPT_DOUBLE("network-timeout", timeout, M_OPT_MIN, .min = 0),
        {0}
    },
    .size = sizeof(struct stream_lavf_params),
    .defaults = &(const struct stream_lavf_params){
        .useragent = (char *)mpv_version,
    },
};

static const char *const http_like[];

