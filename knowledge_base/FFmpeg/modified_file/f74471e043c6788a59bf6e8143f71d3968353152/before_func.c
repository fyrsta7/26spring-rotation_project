    }

    return opt_order;
}


static void encode_residual_verbatim(int32_t *res, int32_t *smp, int n)
{
    assert(n > 0);
    memcpy(res, smp, n * sizeof(int32_t));
}

static void encode_residual_fixed(int32_t *res, const int32_t *smp, int n,
                                  int order)
{
    int i;

