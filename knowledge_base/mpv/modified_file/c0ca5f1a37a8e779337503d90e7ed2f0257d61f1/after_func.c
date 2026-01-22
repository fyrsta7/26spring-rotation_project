{
    struct ao *ao = data;

    MP_WARN(ao, "Error during playback: %s, %s\n", spa_strerror(res), message);
}

static void on_core_info(void *data, const struct pw_core_info *info)
{
    struct ao *ao = data;

    MP_VERBOSE(ao, "Core user: %s\n", info->user_name);
    MP_VERBOSE(ao, "Core host: %s\n", info->host_name);
    MP_VERBOSE(ao, "Core version: %s\n", info->version);
    MP_VERBOSE(ao, "Core name: %s\n", info->name);
}

static const struct pw_core_events core_events = {
    .version = PW_VERSION_CORE_EVENTS,
    .error = on_error,
    .info = on_core_info,
};

static int pipewire_init_boilerplate(struct ao *ao)
{
    struct priv *p = ao->priv;
    struct pw_context *context;

    pw_init(NULL, NULL);

    MP_VERBOSE(ao, "Headers version: %s\n", pw_get_headers_version());
    MP_VERBOSE(ao, "Library version: %s\n", pw_get_library_version());

    p->loop = pw_thread_loop_new("mpv/ao/pipewire", NULL);
    if (p->loop == NULL)
        return -1;

    pw_thread_loop_lock(p->loop);

    if (pw_thread_loop_start(p->loop) < 0)
        goto error;

    context = pw_context_new(
            pw_thread_loop_get_loop(p->loop),
            pw_properties_new(PW_KEY_CONFIG_NAME, "client-rt.conf", NULL),
            0);
    if (!context)
        goto error;

    p->core = pw_context_connect(
            context,
            pw_properties_new(PW_KEY_REMOTE_NAME, p->options.remote, NULL),
            0);
    if (!p->core) {
        MP_MSG(ao, ao->probing ? MSGL_V : MSGL_ERR,
