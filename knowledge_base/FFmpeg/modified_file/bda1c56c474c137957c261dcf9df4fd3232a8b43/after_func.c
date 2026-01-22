
    for(x=0; x<8; x++){
        temp[x      ] = 4*src[x           ];
        temp[x + 7*8] = 4*src[x + 7*stride];
    }
    for(y=1; y<7; y++){
        for(x=0; x<8; x++){
            xy = y * stride + x;
            yz = y * 8 + x;
            temp[yz] = src[xy - stride] + 2*src[xy] + src[xy + stride];
        }
    }
        
    for(y=0; y<8; y++){
        src[  y*stride] = (temp[  y*8] + 2)>>2;
        src[7+y*stride] = (temp[7+y*8] + 2)>>2;
        for(x=1; x<7; x++){
            xy = y * stride + x;
            yz = y * 8 + x;
            src[xy] = (temp[yz-1] + 2*temp[yz] + temp[yz+1] + 8)>>4;
        }
    }
}

static inline void h264_loop_filter_luma_c(uint8_t *pix, int xstride, int ystride, int alpha, int beta, int *tc0)
{
    int i, d;
    for( i = 0; i < 4; i++ ) {
        if( tc0[i] < 0 ) {
            pix += 4*ystride;
            continue;
        }
        for( d = 0; d < 4; d++ ) {
            const int p0 = pix[-1*xstride];
            const int p1 = pix[-2*xstride];
            const int p2 = pix[-3*xstride];
            const int q0 = pix[0];
            const int q1 = pix[1*xstride];
            const int q2 = pix[2*xstride];
    
