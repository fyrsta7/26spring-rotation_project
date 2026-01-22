
    inlink->w = mb->w;
    inlink->h = mb->h;
    inlink->time_base = mb->time_base;

    return 0;
}

static void fill_from_cache(AVFilterContext *ctx, uint32_t *color, int *in_cidx, int *out_cidx, double py, double scale){
    MBContext *mb = ctx->priv;
    for(; *in_cidx < mb->cache_used; (*in_cidx)++){
        Point *p= &mb->point_cache[*in_cidx];
        int x;
        if(*in_cidx >= mb->cache_used || p->p[1] > py)
            break;
        x= round((p->p[0] - mb->start_x) / scale + mb->w/2);
        if(x<0 || x >= mb->w)
            continue;
        if(color) color[x] = p->val;
        if(out_cidx && *out_cidx < mb->cache_allocated)
            mb->next_cache[(*out_cidx)++]= *p;
    }
}

static void draw_mandelbrot(AVFilterContext *ctx, uint32_t *color, int linesize, int64_t pts)
{
    MBContext *mb = ctx->priv;
    int x,y,i, in_cidx=0, next_cidx=0, tmp_cidx;
    double scale= mb->start_scale*pow(mb->end_scale/mb->start_scale, pts/mb->end_pts);
    int use_zyklus=0;
    fill_from_cache(ctx, NULL, &in_cidx, NULL, mb->start_y+scale*(-mb->h/2-0.5), scale);
    for(y=0; y<mb->h; y++){
        const double ci=mb->start_y+scale*(y-mb->h/2);
        memset(color+linesize*y, 0, sizeof(*color)*mb->w);
        fill_from_cache(ctx, color+linesize*y, &in_cidx, &next_cidx, ci, scale);
        tmp_cidx= in_cidx;
        fill_from_cache(ctx, color+linesize*y, &tmp_cidx, NULL, ci + scale/2, scale);

        for(x=0; x<mb->w; x++){
            const double cr=mb->start_x+scale*(x-mb->w/2);
            double zr=cr;
            double zi=ci;
            uint32_t c=0;

            if(color[x + y*linesize] & 0xFF000000)
                continue;

            use_zyklus= (x==0 || color[x-1 + y*linesize] == 0xFF000000);

            for(i=0; i<mb->maxiter; i++){
                double t;
                if(zr*zr + zi*zi > mb->bailout){
                    switch(mb->outer){
                    case            ITERATION_COUNT: zr= i; break;
                    case NORMALIZED_ITERATION_COUNT: zr= i + log2(log(mb->bailout) / log(zr*zr + zi*zi)); break;
                    }
                    c= lrintf((sin(zr)+1)*127) + lrintf((sin(zr/1.234)+1)*127)*256*256 + lrintf((sin(zr/100)+1)*127)*256;
                    break;
                }
                t= zr*zr - zi*zi + cr;
                zi= 2*zr*zi + ci;
                if(use_zyklus){
                    mb->zyklus[i][0]= t;
                    mb->zyklus[i][1]= zi;
                }
                i++;
                if(t*t + zi*zi > mb->bailout){
                    switch(mb->outer){
                    case            ITERATION_COUNT: zr= i; break;
                    case NORMALIZED_ITERATION_COUNT: zr= i + log2(log(mb->bailout) / log(t*t + zi*zi)); break;
                    }
                    c= lrintf((sin(zr)+1)*127) + lrintf((sin(zr/1.234)+1)*127)*256*256 + lrintf((sin(zr/100)+1)*127)*256;
                    break;
                }
