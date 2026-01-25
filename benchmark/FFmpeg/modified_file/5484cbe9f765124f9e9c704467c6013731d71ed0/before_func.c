static void fill_from_cache(AVFilterContext *ctx, uint32_t *color, int *in_cidx, int *out_cidx, double py, double scale){
    MBContext *s = ctx->priv;
    if(s->morphamp)
        return;
    for(; *in_cidx < s->cache_used; (*in_cidx)++){
        Point *p= &s->point_cache[*in_cidx];
        int x;
        if(p->p[1] > py)
            break;
        x= round((p->p[0] - s->start_x) / scale + s->w/2);
        if(x<0 || x >= s->w)
            continue;
        if(color) color[x] = p->val;
        if(out_cidx && *out_cidx < s->cache_allocated)
            s->next_cache[(*out_cidx)++]= *p;
    }
}