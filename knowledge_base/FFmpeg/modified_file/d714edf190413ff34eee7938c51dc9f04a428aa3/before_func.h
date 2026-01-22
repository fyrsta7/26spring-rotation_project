    int mod;
} Ticker;

extern void ticker_init(Ticker *tick, INT64 inrate, INT64 outrate);

static inline int ticker_tick(Ticker *tick, int num)
{
    int n = num * tick->div;

    tick->value += num * tick->mod;
    while (tick->value > 0) {
        tick->value -= tick->inrate;
