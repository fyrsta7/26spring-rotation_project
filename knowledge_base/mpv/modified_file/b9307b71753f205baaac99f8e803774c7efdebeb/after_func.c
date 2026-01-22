    atomic_store(&c->triggered, false);
#ifdef __MINGW32__
    ResetEvent(c->event);
#endif
