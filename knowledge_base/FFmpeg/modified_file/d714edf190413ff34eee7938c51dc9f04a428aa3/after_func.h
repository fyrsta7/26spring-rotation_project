    int mod;
} Ticker;

extern void ticker_init(Ticker *tick, INT64 inrate, INT64 outrate);

static inline int ticker_tick(Ticker *tick, int num)
{
    int n = num * tick->div;

    tick->value += num * tick->mod;
#if 1
    if (tick->value > 0) {
        n += (tick->value / tick->inrate);
        tick->value = tick->value % tick->inrate;
        if (tick->value > 0) {
            tick->value -= tick->inrate;
            n++;
        }
    }
#else
    while (tick->value > 0) {
        tick->value -= tick->inrate;
