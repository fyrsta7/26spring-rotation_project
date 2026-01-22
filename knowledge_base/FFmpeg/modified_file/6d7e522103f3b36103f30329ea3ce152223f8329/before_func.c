    uint8_t r = 0, g = 0, b = 0;

    if (bytestream2_get_bytes_left(gbyte) < 3 * avctx->width * avctx->height)
        return AVERROR_INVALIDDATA;

    for (int y = 0; y < avctx->height; y++) {
        for (int x = 0; x < avctx->width; x++) {
            dst[x*3+0] = bytestream2_get_byteu(gbyte) + r;
            r = dst[x*3+0];
            dst[x*3+1] = bytestream2_get_byteu(gbyte) + g;
            g = dst[x*3+1];
            dst[x*3+2] = bytestream2_get_byteu(gbyte) + b;
            b = dst[x*3+2];
        }
        dst -= frame->linesize[0];
    }

    return 0;
}

static int fill_pixels(uint8_t **y0, uint8_t **y1,
                       uint8_t **u, uint8_t **v,
                       int ylinesize, int ulinesize, int vlinesize,
                       uint8_t *fill,
                       int *nx, int *ny, int *np, int w, int h)
{
    uint8_t *y0dst = *y0;
    uint8_t *y1dst = *y1;
    uint8_t *udst = *u;
    uint8_t *vdst = *v;
    int x = *nx, y = *ny, pos = *np;

    if (pos == 0) {
        y0dst[2*x+0] += fill[0];
        y0dst[2*x+1] += fill[1];
        y1dst[2*x+0] += fill[2];
        y1dst[2*x+1] += fill[3];
        pos++;
    } else if (pos == 1) {
        udst[x] += fill[0];
        vdst[x] += fill[1];
        x++;
        if (x >= w) {
            x = 0;
            y++;
            if (y >= h)
                return 1;
            y0dst -= 2*ylinesize;
            y1dst -= 2*ylinesize;
            udst  -=   ulinesize;
            vdst  -=   vlinesize;
        }
        y0dst[2*x+0] += fill[2];
        y0dst[2*x+1] += fill[3];
        pos++;
    } else if (pos == 2) {
        y1dst[2*x+0] += fill[0];
        y1dst[2*x+1] += fill[1];
        udst[x]      += fill[2];
        vdst[x]      += fill[3];
        x++;
        if (x >= w) {
            x = 0;
            y++;
