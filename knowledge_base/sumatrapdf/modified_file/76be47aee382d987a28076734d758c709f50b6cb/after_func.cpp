
static RenderedBitmap *new_rendered_fz_pixmap(fz_context *ctx, fz_pixmap *pixmap)
{
    int paletteSize = 0;
    bool hasPalette = false;

    int w = pixmap->w;
    int h = pixmap->h;
    int rows8 = ((w + 3) / 4) * 4;

    ScopedMem<BITMAPINFO> bmi((BITMAPINFO *)calloc(1, sizeof(BITMAPINFO) + 255 * sizeof(RGBQUAD)));

    // always try to produce an 8-bit palette for saving some memory
    unsigned char *bmpData = (unsigned char *)calloc(rows8, h);
    if (!bmpData)
        return NULL;
    fz_pixmap *bgrPixmap = NULL;
    if (bmpData && pixmap->n == 4 &&
        pixmap->colorspace == fz_device_rgb(ctx)) {
        unsigned char *dest = bmpData;
        unsigned char *source = pixmap->samples;
        uint32_t *palette = (uint32_t *)bmi.Get()->bmiColors;
        BYTE grayIdxs[256] = { 0 };

        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                RGBQUAD c;

                c.rgbRed = *source++;
                c.rgbGreen = *source++;
                c.rgbBlue = *source++;
                c.rgbReserved = 0;
                source++;

                /* find this color in the palette */
                int k;
                bool isGray = c.rgbRed == c.rgbGreen && c.rgbRed == c.rgbBlue;
                if (isGray) {
                    k = grayIdxs[c.rgbRed] || palette[0] == *(uint32_t *)&c ? grayIdxs[c.rgbRed] : paletteSize;
                }
                else {
                    for (k = 0; k < paletteSize && palette[k] != *(uint32_t *)&c; k++);
                }
                /* add it to the palette if it isn't in there and if there's still space left */
                if (k == paletteSize) {
                    if (++paletteSize > 256)
                        goto ProducingPaletteDone;
                    if (isGray)
                        grayIdxs[c.rgbRed] = (BYTE)k;
                    palette[k] = *(uint32_t *)&c;
                }
                /* 8-bit data consists of indices into the color palette */
                *dest++ = k;
            }
            dest += rows8 - w;
        }
ProducingPaletteDone:
        hasPalette = paletteSize <= 256;
    }
    if (!hasPalette) {
        free(bmpData);
        /* BGRA is a GDI compatible format */
        fz_try(ctx) {
            fz_irect bbox;
            fz_colorspace *colorspace = fz_device_bgr(ctx);
            bgrPixmap = fz_new_pixmap_with_bbox(ctx, colorspace, fz_pixmap_bbox(ctx, pixmap, &bbox));
            fz_convert_pixmap(ctx, bgrPixmap, pixmap);
        }
        fz_catch(ctx) {
            return NULL;
        }
    }
    AssertCrash(hasPalette || bgrPixmap);

    BITMAPINFOHEADER *bmih = &bmi.Get()->bmiHeader;
    bmih->biSize = sizeof(*bmih);
    bmih->biWidth = w;
    bmih->biHeight = -h;
    bmih->biPlanes = 1;
    bmih->biCompression = BI_RGB;
    bmih->biBitCount = hasPalette ? 8 : 32;
    bmih->biSizeImage = h * (hasPalette ? rows8 : w * 4);
    bmih->biClrUsed = hasPalette ? paletteSize : 0;

    void *data = NULL;
    HANDLE hMap = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, bmih->biSizeImage, NULL);
    HBITMAP hbmp = CreateDIBSection(NULL, bmi, DIB_RGB_COLORS, &data, hMap, 0);
    if (hbmp)
        memcpy(data, hasPalette ? bmpData : bgrPixmap->samples, bmih->biSizeImage);

    if (hasPalette)
        free(bmpData);
    else
        fz_drop_pixmap(ctx, bgrPixmap);

    // return a RenderedBitmap even if hbmp is NULL so that callers can
    // distinguish rendering errors from GDI resource exhaustion
    // (and in the latter case retry using smaller target rectangles)
    return new RenderedBitmap(hbmp, SizeI(w, h), hMap);
