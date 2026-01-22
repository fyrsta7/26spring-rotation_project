    float *sp_a = s->sofa.sp_a; /* azimuth angle */
    float *sp_e = s->sofa.sp_e; /* elevation angle */
    float *sp_r = s->sofa.sp_r; /* radius */
    int m_dim = s->sofa.m_dim; /* no. measurements */
    int best_id = 0; /* index m currently closest to desired source pos. */
    float delta = 1000; /* offset between desired and currently best pos. */
    float current;
    int i;

    for (i = 0; i < m_dim; i++) {
        /* search through all measurements in currently selected SOFA file */
        /* distance of current to desired source position: */
        current = fabs(sp_a[i] - azim) +
                  fabs(sp_e[i] - elev) +
                  fabs(sp_r[i] - radius);
        if (current <= delta) {
            /* if current distance is smaller than smallest distance so far */
            delta = current;
            best_id = i; /* remember index */
        }
    }

    return best_id;
}

static int compensate_volume(AVFilterContext *ctx)
{
    struct SOFAlizerContext *s = ctx->priv;
    float compensate;
