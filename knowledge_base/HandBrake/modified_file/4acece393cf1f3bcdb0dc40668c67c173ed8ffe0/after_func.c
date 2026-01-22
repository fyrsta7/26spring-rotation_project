            color_space = SWS_CS_SMPTE240M;
            break;
        case HB_COLR_MAT_BT709:
            color_space = SWS_CS_ITU709;
            break;
        default:
            break;
    }

    return color_space;
}

struct SwsContext*
hb_sws_get_context(int srcW, int srcH, enum AVPixelFormat srcFormat,
                   int dstW, int dstH, enum AVPixelFormat dstFormat,
                   int flags, int colorspace)
{
    struct SwsContext * ctx;

    ctx = sws_alloc_context();
    if ( ctx )
    {
        int srcRange, dstRange;

        srcRange = handle_jpeg(&srcFormat);
        dstRange = handle_jpeg(&dstFormat);
        flags |= SWS_FULL_CHR_H_INT | SWS_FULL_CHR_H_INP;

        av_opt_set_int(ctx, "srcw", srcW, 0);
        av_opt_set_int(ctx, "srch", srcH, 0);
        av_opt_set_int(ctx, "src_range", srcRange, 0);
        av_opt_set_int(ctx, "src_format", srcFormat, 0);
        av_opt_set_int(ctx, "dstw", dstW, 0);
        av_opt_set_int(ctx, "dsth", dstH, 0);
        av_opt_set_int(ctx, "dst_range", dstRange, 0);
        av_opt_set_int(ctx, "dst_format", dstFormat, 0);
        av_opt_set_int(ctx, "sws_flags", flags, 0);

        sws_setColorspaceDetails( ctx,
                      sws_getCoefficients( colorspace ), // src colorspace
                      srcRange, // src range 0 = MPG, 1 = JPG
                      sws_getCoefficients( colorspace ), // dst colorspace
                      dstRange, // dst range 0 = MPG, 1 = JPG
