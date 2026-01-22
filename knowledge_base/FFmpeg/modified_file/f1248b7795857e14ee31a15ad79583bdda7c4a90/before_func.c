                                       const uint8_t *src, int linesize, int offx, int offy,
                                       int e, int w, int h)
{
    // ii has a surrounding padding of thickness "e"
    const int ii_w = w + e*2;
    const int ii_h = h + e*2;

    // we center the first source
    const int s1x = e;
    const int s1y = e;

    // 2nd source is the frame with offsetting
    const int s2x = e + offx;
    const int s2y = e + offy;

    // get the dimension of the overlapping rectangle where it is always safe
    // to compare the 2 sources pixels
    const int startx_safe = FFMAX(s1x, s2x);
    const int starty_safe = FFMAX(s1y, s2y);
    const int endx_safe   = FFMIN(s1x + w, s2x + w);
    const int endy_safe   = FFMIN(s1y + h, s2y + h);

    // top part where only one of s1 and s2 is still readable, or none at all
    compute_unsafe_ssd_integral_image(ii, ii_linesize_32,
                                      0, 0,
                                      src, linesize,
                                      offx, offy, e, w, h,
