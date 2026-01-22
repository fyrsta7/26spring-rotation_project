        c= color[(x-1) + (y-1)*linesize];
        a= color[(x-1) + (y+0)*linesize];
        b= color[(x+1) + (y-1)*linesize];
    }else
        return 0;

    for(i=0; i<3; i++){
        int s= 8*i;
        uint8_t ac= a>>s;
        uint8_t bc= b>>s;
        uint8_t cc= c>>s;
        uint8_t dc= d>>s;
        int ipolab= (ac + bc);
        int ipolcd= (cc + dc);
        if(FFABS(ipolab - ipolcd) > 5)
            return 0;
        if(FFABS(ac-bc)+FFABS(cc-dc) > 20)
            return 0;
        ipol |= ((ipolab + ipolcd + 2)/4)<<s;
    }
    color[x + y*linesize]= ipol;
    return 1;
}

static void draw_mandelbrot(AVFilterContext *ctx, uint32_t *color, int linesize, int64_t pts)
{
    MBContext *s = ctx->priv;
    int x,y,i, in_cidx=0, next_cidx=0, tmp_cidx;
    double scale= s->start_scale*pow(s->end_scale/s->start_scale, pts/s->end_pts);
    int use_zyklus=0;
    fill_from_cache(ctx, NULL, &in_cidx, NULL, s->start_y+scale*(-s->h/2-0.5), scale);
    tmp_cidx= in_cidx;
    memset(color, 0, sizeof(*color)*s->w);
    for(y=0; y<s->h; y++){
        int y1= y+1;
        const double ci=s->start_y+scale*(y-s->h/2);
        fill_from_cache(ctx, NULL, &in_cidx, &next_cidx, ci, scale);
        if(y1<s->h){
            memset(color+linesize*y1, 0, sizeof(*color)*s->w);
            fill_from_cache(ctx, color+linesize*y1, &tmp_cidx, NULL, ci + 3*scale/2, scale);
        }

        for(x=0; x<s->w; x++){
            float av_uninit(epsilon);
            const double cr=s->start_x+scale*(x-s->w/2);
            double zr=cr;
            double zi=ci;
            uint32_t c=0;
            double dv= s->dither / (double)(1LL<<32);
            s->dither= s->dither*1664525+1013904223;

            if(color[x + y*linesize] & 0xFF000000)
                continue;
            if(!s->morphamp){
                if(interpol(s, color, x, y, linesize)){
                    if(next_cidx < s->cache_allocated){
                        s->next_cache[next_cidx  ].p[0]= cr;
                        s->next_cache[next_cidx  ].p[1]= ci;
                        s->next_cache[next_cidx++].val = color[x + y*linesize];
                    }
                    continue;
                }
            }else{
                zr += cos(pts * s->morphxf) * s->morphamp;
                zi += sin(pts * s->morphyf) * s->morphamp;
            }

            use_zyklus= (x==0 || s->inner!=BLACK ||color[x-1 + y*linesize] == 0xFF000000);
            if(use_zyklus)
                epsilon= scale*(abs(x-s->w/2) + abs(y-s->h/2))/s->w;

#define Z_Z2_C(outr,outi,inr,ini)\
            outr= inr*inr - ini*ini + cr;\
            outi= 2*inr*ini + ci;

#define Z_Z2_C_ZYKLUS(outr,outi,inr,ini, Z)\
            Z_Z2_C(outr,outi,inr,ini)\
            if(use_zyklus){\
                if(Z && fabs(s->zyklus[i>>1][0]-outr)+fabs(s->zyklus[i>>1][1]-outi) <= epsilon)\
                    break;\
            }\
            s->zyklus[i][0]= outr;\
            s->zyklus[i][1]= outi;\



            for(i=0; i<s->maxiter-8; i++){
                double t;
                Z_Z2_C_ZYKLUS(t, zi, zr, zi, 0)
                i++;
                Z_Z2_C_ZYKLUS(zr, zi, t, zi, 1)
                i++;
                Z_Z2_C_ZYKLUS(t, zi, zr, zi, 0)
                i++;
                Z_Z2_C_ZYKLUS(zr, zi, t, zi, 1)
                i++;
                Z_Z2_C_ZYKLUS(t, zi, zr, zi, 0)
                i++;
                Z_Z2_C_ZYKLUS(zr, zi, t, zi, 1)
                i++;
                Z_Z2_C_ZYKLUS(t, zi, zr, zi, 0)
                i++;
                Z_Z2_C_ZYKLUS(zr, zi, t, zi, 1)
                if(zr*zr + zi*zi > s->bailout){
                    i-= FFMIN(7, i);
                    for(; i<s->maxiter; i++){
                        zr= s->zyklus[i][0];
                        zi= s->zyklus[i][1];
                        if(zr*zr + zi*zi > s->bailout){
                            switch(s->outer){
                            case            ITERATION_COUNT:
                                zr = i;
                                c = lrintf((sinf(zr)+1)*127) + lrintf((sinf(zr/1.234)+1)*127)*256*256 + lrintf((sinf(zr/100)+1)*127)*256;
                                break;
                            case NORMALIZED_ITERATION_COUNT:
                                zr = i + log2(log(s->bailout) / log(zr*zr + zi*zi));
                                c = lrintf((sinf(zr)+1)*127) + lrintf((sinf(zr/1.234)+1)*127)*256*256 + lrintf((sinf(zr/100)+1)*127)*256;
                                break;
                            case                      WHITE:
                                c = 0xFFFFFF;
                                break;
                            case                      OUTZ:
                                zr /= s->bailout;
                                zi /= s->bailout;
                                c = (((int)(zr*128+128))&0xFF)*256 + (((int)(zi*128+128))&0xFF);
                            }
                            break;
                        }
                    }
                    break;
                }
            }
            if(!c){
                if(s->inner==PERIOD){
                    int j;
                    for(j=i-1; j; j--)
                        if(SQR(s->zyklus[j][0]-zr) + SQR(s->zyklus[j][1]-zi) < epsilon*epsilon*10)
                            break;
                    if(j){
                        c= i-j;
                        c= ((c<<5)&0xE0) + ((c<<10)&0xE000) + ((c<<15)&0xE00000);
                    }
                }else if(s->inner==CONVTIME){
                    c= floor(i*255.0/s->maxiter+dv)*0x010101;
                } else if(s->inner==MINCOL){
                    int j;
                    double closest=9999;
                    int closest_index=0;
