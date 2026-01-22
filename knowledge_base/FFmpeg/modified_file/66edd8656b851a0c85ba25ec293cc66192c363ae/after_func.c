
static int estimate_best_order(double *ref, int min_order, int max_order)
{
    int i, est;

    est = min_order;
    for(i=max_order-1; i>=min_order-1; i--) {
        if(ref[i] > 0.10) {
            est = i+1;
            break;
        }
    }
    return est;
}

int ff_lpc_calc_ref_coefs(LPCContext *s,
                          const int32_t *samples, int order, double *ref)
{
    double autoc[MAX_LPC_ORDER + 1];

    s->lpc_apply_welch_window(samples, s->blocksize, s->windowed_samples);
    s->lpc_compute_autocorr(s->windowed_samples, s->blocksize, order, autoc);
