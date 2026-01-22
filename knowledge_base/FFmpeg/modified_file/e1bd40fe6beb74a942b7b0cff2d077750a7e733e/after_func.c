    return sum_ ## suf();                                               \
}                                                                       \
                                                                        \
static int sad16_xy2_ ## suf(MpegEncContext *v, uint8_t *blk2,          \
                             uint8_t *blk1, int stride, int h)          \
{                                                                       \
    __asm__ volatile (                                                  \
        "pxor %%mm7, %%mm7     \n\t"                                    \
        "pxor %%mm6, %%mm6     \n\t"                                    \
        ::);                                                            \
                                                                        \
    sad8_4_ ## suf(blk1,     blk2,     stride, h);                      \
    sad8_4_ ## suf(blk1 + 8, blk2 + 8, stride, h);                      \
                                                                        \
    return sum_ ## suf();                                               \
}                                                                       \

PIX_SAD(mmx)
PIX_SAD(mmxext)

#endif /* HAVE_INLINE_ASM */

av_cold void ff_dsputil_init_pix_mmx(DSPContext *c, AVCodecContext *avctx)
{
#if HAVE_INLINE_ASM
    int cpu_flags = av_get_cpu_flags();

    if (INLINE_MMX(cpu_flags)) {
        c->pix_abs[0][0] = sad16_mmx;
        c->pix_abs[0][1] = sad16_x2_mmx;
        c->pix_abs[0][2] = sad16_y2_mmx;
        c->pix_abs[0][3] = sad16_xy2_mmx;
        c->pix_abs[1][0] = sad8_mmx;
        c->pix_abs[1][1] = sad8_x2_mmx;
        c->pix_abs[1][2] = sad8_y2_mmx;
        c->pix_abs[1][3] = sad8_xy2_mmx;

        c->sad[0] = sad16_mmx;
        c->sad[1] = sad8_mmx;
