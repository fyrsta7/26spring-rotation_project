            : : "m" (src[0]), "m" (src[1]));

        asm volatile(
            /* mm2 += C * src[0..3] + D * src[1..4] */
            "movq %%mm0, %%mm3\n\t"
            "movq %%mm1, %%mm4\n\t"
            "pmullw %%mm6, %%mm3\n\t"
            "pmullw %0, %%mm4\n\t"
            "paddw %%mm3, %%mm2\n\t"
            "paddw %%mm4, %%mm2\n\t"
            : : "m" (DD));

        asm volatile(
            /* dst[0..3] = pack((mm2 + 32) >> 6) */
            "paddw %1, %%mm2\n\t"
            "psrlw $6, %%mm2\n\t"
            "packuswb %%mm7, %%mm2\n\t"
            H264_CHROMA_OP4(%0, %%mm2, %%mm3)
            "movd %%mm2, %0\n\t"
            : "=m" (dst[0]) : "m" (ff_pw_32));
        dst += stride;
    }
}

#ifdef H264_CHROMA_MC2_TMPL
static void H264_CHROMA_MC2_TMPL(uint8_t *dst/*align 2*/, uint8_t *src/*align 1*/, int stride, int h, int x, int y)
{
    int tmp = ((1<<16)-1)*x + 8;
    int CD= tmp*y;
    int AB= (tmp<<3) - CD;
    asm volatile(
        /* mm5 = {A,B,A,B} */
        /* mm6 = {C,D,C,D} */
        "movd %0, %%mm5\n\t"
        "movd %1, %%mm6\n\t"
        "punpckldq %%mm5, %%mm5\n\t"
        "punpckldq %%mm6, %%mm6\n\t"
        "pxor %%mm7, %%mm7\n\t"
        /* mm0 = src[0,1,1,2] */
        "movd %2, %%mm0\n\t"
        "punpcklbw %%mm7, %%mm0\n\t"
        "pshufw $0x94, %%mm0, %%mm0\n\t"
        :: "r"(AB), "r"(CD), "m"(src[0]));


    asm volatile(
        "1:\n\t"
        "addl %4, %1\n\t"
