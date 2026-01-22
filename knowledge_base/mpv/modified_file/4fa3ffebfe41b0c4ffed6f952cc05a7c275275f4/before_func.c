{
    struct ao_push_state *p = ao->api_priv;

    pthread_mutex_lock(&p->lock);
    p->final_chunk = true;
    p->drain = true;
    wakeup_playthread(ao);
    while (p->drain)
        pthread_cond_wait(&p->wakeup_drain, &p->lock);
    pthread_mutex_unlock(&p->lock);

    if (!ao->driver->drain)
        ao_wait_drain(ao);
}

static int unlocked_get_space(struct ao *ao)
{
    struct ao_push_state *p = ao->api_priv;
    int space = mp_audio_buffer_get_write_available(p->buffer);
    if (ao->driver->get_space) {
