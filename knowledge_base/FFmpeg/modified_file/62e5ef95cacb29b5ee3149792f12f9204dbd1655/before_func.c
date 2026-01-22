            b = (i&1     )*255;
            break;
        case PIX_FMT_BGR4_BYTE:
            b = (i>>3    )*255;
            g = ((i>>1)&3)*85;
            r = (i&1     )*255;
            break;
        case PIX_FMT_GRAY8:
            r = b = g = i;
            break;
        default:
            return AVERROR(EINVAL);
        }
        pal[i] = b + (g<<8) + (r<<16) + (0xFF<<24);
    }

    return 0;
}

int av_image_alloc(uint8_t *pointers[4], int linesizes[4],
                   int w, int h, enum PixelFormat pix_fmt, int align)
{
    int i, ret;
    uint8_t *buf;

    if ((ret = av_image_check_size(w, h, 0, NULL)) < 0)
        return ret;
    if ((ret = av_image_fill_linesizes(linesizes, pix_fmt, w)) < 0)
        return ret;
