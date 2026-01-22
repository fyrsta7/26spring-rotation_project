        "bswap %%edx                            \n\t"
        "shrl $15, %%edx                        \n\t"
        "addl $2, %%ebx                         \n\t"
        "addl %%edx, %%eax                      \n\t"
        "movl %%ebx, "BYTE     "(%1)            \n\t"
        "1:                                     \n\t"
        "movl %%eax, "LOW      "(%1)            \n\t"

        :"+c"(val)
        :"r"(c)
        : "%eax", "%ebx", "%edx", "memory"
    );
    return val;
#else
    int range, mask;
    c->low += c->low;

    if(!(c->low & CABAC_MASK))
        refill(c);

    range= c->range<<17;
    c->low -= range;
    mask= c->low >> 31;
    range &= mask;
    c->low += range;
    return (val^mask)-mask;
#endif
}

//FIXME the x86 code from this file should be moved into i386/h264 or cabac something.c/h (note ill kill you if you move my code away from under my fingers before iam finished with it!)
//FIXME use some macros to avoid duplicatin get_cabac (cant be done yet as that would make optimization work hard)
#ifdef ARCH_X86
static int decode_significance_x86(CABACContext *c, int max_coeff, uint8_t *significant_coeff_ctx_base, int *index){
    void *end= significant_coeff_ctx_base + max_coeff - 1;
    int minusstart= -(int)significant_coeff_ctx_base;
    int minusindex= 4-(int)index;
    int coeff_count;
    asm volatile(
        "movl "RANGE    "(%3), %%esi            \n\t"
        "movl "LOW      "(%3), %%ebx            \n\t"

        "2:                                     \n\t"

        BRANCHLESS_GET_CABAC("%%edx", "%3", "(%1)", "%%ebx", "%%bx", "%%esi", "%%eax", "%%al")

        "test $1, %%edx                         \n\t"
        " jz 3f                                 \n\t"

        BRANCHLESS_GET_CABAC("%%edx", "%3", "61(%1)", "%%ebx", "%%bx", "%%esi", "%%eax", "%%al")
