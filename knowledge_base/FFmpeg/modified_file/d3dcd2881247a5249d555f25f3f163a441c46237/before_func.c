static void calc_sum_top(int pmax, uint32_t *data, int n, int pred_order,
                         uint64_t sums[MAX_PARTITIONS])
{
    int i;
    int parts;
    uint32_t *res, *res_end;

    /* sums for highest level */
    parts   = (1 << pmax);
    res     = &data[pred_order];
    res_end = &data[n >> pmax];
    for (i = 0; i < parts; i++) {
        uint64_t sum = 0;
        while (res < res_end)
            sum += *(res++);
        sums[i] = sum;
        res_end += n >> pmax;
    }
}

static void calc_sum_next(int level, uint64_t sums[MAX_PARTITIONS])
{
    int i;
    int parts = (1 << level);
    for (i = 0; i < parts; i++)
        sums[i] = sums[2*i] + sums[2*i+1];
}

static uint64_t calc_rice_params(RiceContext *rc, int pmin, int pmax,
                                 int32_t *data, int n, int pred_order)
{
    int i;
    uint64_t bits[MAX_PARTITION_ORDER+1];
    int opt_porder;
    RiceContext tmp_rc;
