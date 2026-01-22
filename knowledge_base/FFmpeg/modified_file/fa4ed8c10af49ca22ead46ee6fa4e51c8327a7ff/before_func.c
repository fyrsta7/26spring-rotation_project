    return porder;
}

static uint32_t calc_rice_params_fixed(RiceContext *rc, int pmin, int pmax,
                                       int32_t *data, int n, int pred_order,
                                       int bps)
{
    uint32_t bits;
    pmin = get_max_p_order(pmin, n, pred_order);
    pmax = get_max_p_order(pmax, n, pred_order);
    bits = pred_order*bps + 6;
    bits += calc_rice_params(rc, pmin, pmax, data, n, pred_order);
    return bits;
}

