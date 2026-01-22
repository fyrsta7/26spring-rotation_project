    asm(
        "cmp $0x2000000, %0         \n\t"
        "sbb %%edx, %%edx           \n\t"
        "mov %0, %%eax              \n\t"
        "and %%edx, %0              \n\t"
        "and %1, %%edx              \n\t"
        "add %%eax, %0              \n\t"
        "add %%edx, %1              \n\t"
        : "+r"(c->range), "+r"(c->low), "+a"(temp), "+d"(temp2)
    );
#else
    int temp2;
    //P3:677    athlon:511
    asm(
        "cmp $0x2000000, %0         \n\t"
        "lea (%0, %0), %%eax        \n\t"
        "lea (%1, %1), %%edx        \n\t"
        "cmovb %%eax, %0            \n\t"
        "cmovb %%edx, %1            \n\t"
        : "+r"(c->range), "+r"(c->low), "+a"(temp), "+d"(temp2)
    );
#endif
#else
    //P3:675    athlon:476
    int shift= (uint32_t)(c->range - (0x200 << CABAC_BITS))>>31;
    c->range<<= shift;
    c->low  <<= shift;
#endif
    if(!(c->low & CABAC_MASK))
        refill(c);
}

static int get_cabac(CABACContext *c, uint8_t * const state){
    //FIXME gcc generates duplicate load/stores for c->low and c->range
#ifdef ARCH_X86
    int bit;

#define LOW          "0"
#define RANGE        "4"
#define LPS_RANGE   "12"
#define LPS_STATE   "12+2*65*4"
#define MPS_STATE   "12+2*65*4+2*64"
#define BYTESTART   "12+2*65*4+4*64"
#define BYTE        "16+2*65*4+4*64"
#define BYTEEND     "20+2*65*4+4*64"
#ifndef BRANCHLESS_CABAD
    asm volatile(
        "movzbl (%1), %%eax                     \n\t"
        "movl "RANGE    "(%2), %%ebx            \n\t"
        "movl "RANGE    "(%2), %%edx            \n\t"
        "shrl $23, %%ebx                        \n\t"
        "leal "LPS_RANGE"(%2, %%eax, 4), %%esi  \n\t"
        "movzbl (%%ebx, %%esi), %%esi           \n\t"
        "shll $17, %%esi                        \n\t"
        "movl "LOW      "(%2), %%ebx            \n\t"
//eax:state ebx:low, edx:range, esi:RangeLPS
        "subl %%esi, %%edx                      \n\t"
        "cmpl %%edx, %%ebx                      \n\t"
        " ja 1f                                 \n\t"
        "cmp $0x2000000, %%edx                  \n\t" //FIXME avoidable
        "setb %%cl                              \n\t"
        "shl %%cl, %%edx                        \n\t"
        "shl %%cl, %%ebx                        \n\t"
        "movzbl "MPS_STATE"(%2, %%eax), %%ecx   \n\t"
        "movb %%cl, (%1)                        \n\t"
//eax:state ebx:low, edx:range, esi:RangeLPS
        "test %%bx, %%bx                        \n\t"
        " jnz 2f                                \n\t"
        "movl "BYTE     "(%2), %%esi            \n\t"
        "subl $0xFFFF, %%ebx                    \n\t"
        "movzwl (%%esi), %%ecx                  \n\t"
        "bswap %%ecx                            \n\t"
        "shrl $15, %%ecx                        \n\t"
        "addl $2, %%esi                         \n\t"
        "addl %%ecx, %%ebx                      \n\t"
        "movl %%esi, "BYTE    "(%2)             \n\t"
        "jmp 2f                                 \n\t"
        "1:                                     \n\t"
//eax:state ebx:low, edx:range, esi:RangeLPS
        "subl %%edx, %%ebx                      \n\t"
        "movl %%esi, %%edx                      \n\t"
        "shr $19, %%esi                         \n\t"
        "movzbl " MANGLE(ff_h264_norm_shift) "(%%esi), %%ecx   \n\t"
        "shll %%cl, %%ebx                       \n\t"
        "shll %%cl, %%edx                       \n\t"
        "movzbl "LPS_STATE"(%2, %%eax), %%ecx   \n\t"
        "movb %%cl, (%1)                        \n\t"
        "addl $1, %%eax                         \n\t"
        "test %%bx, %%bx                        \n\t"
        " jnz 2f                                \n\t"

        "movl "BYTE     "(%2), %%ecx            \n\t"
        "movzwl (%%ecx), %%esi                  \n\t"
        "bswap %%esi                            \n\t"
        "shrl $15, %%esi                        \n\t"
        "subl $0xFFFF, %%esi                    \n\t"
        "addl $2, %%ecx                         \n\t"
        "movl %%ecx, "BYTE    "(%2)             \n\t"

        "leal -1(%%ebx), %%ecx                  \n\t"
        "xorl %%ebx, %%ecx                      \n\t"
        "shrl $17, %%ecx                        \n\t"
        "movzbl " MANGLE(ff_h264_norm_shift) "(%%ecx), %%ecx   \n\t"
        "neg %%cl                               \n\t"
        "add $7, %%cl                           \n\t"

        "shll %%cl , %%esi                      \n\t"
        "addl %%esi, %%ebx                      \n\t"
        "2:                                     \n\t"
        "movl %%edx, "RANGE    "(%2)            \n\t"
        "movl %%ebx, "LOW      "(%2)            \n\t"
        "andl $1, %%eax                         \n\t"

        :"=&a"(bit) //FIXME this is fragile gcc either runs out of registers or misscompiles it (for example if "+a"(bit) or "+m"(*state) is used
        :"r"(state), "r"(c)
        : "%ecx", "%ebx", "%edx", "%esi"
    );
#else
    asm volatile(
        "movzbl (%1), %%eax                     \n\t"
        "movl "RANGE    "(%2), %%ebx            \n\t"
        "movl "RANGE    "(%2), %%edx            \n\t"
        "shrl $23, %%ebx                        \n\t"
        "leal "LPS_RANGE"(%2, %%eax, 4), %%esi  \n\t"
        "movzbl (%%ebx, %%esi), %%esi           \n\t"
        "shll $17, %%esi                        \n\t"
        "movl "LOW      "(%2), %%ebx            \n\t"
//eax:state ebx:low, edx:range, esi:RangeLPS
        "subl %%esi, %%edx                      \n\t"
#ifdef CMOV_IS_FAST //FIXME actually define this somewhere
        "cmpl %%ebx, %%edx                      \n\t"
        "cmova %%edx, %%esi                     \n\t"
        "sbbl %%ecx, %%ecx                      \n\t"
        "andl %%ecx, %%edx                      \n\t"
        "subl %%edx, %%ebx                      \n\t"
        "xorl %%ecx, %%eax                      \n\t"
#else
        "movl %%edx, %%ecx                      \n\t"
        "subl %%ebx, %%edx                      \n\t"
        "sarl $31, %%edx                        \n\t" //lps_mask
        "subl %%ecx, %%esi                      \n\t" //RangeLPS - range
        "andl %%edx, %%esi                      \n\t" //(RangeLPS - range)&lps_mask
        "addl %%ecx, %%esi                      \n\t" //new range
        "andl %%edx, %%ecx                      \n\t"
        "subl %%ecx, %%ebx                      \n\t"
        "xorl %%edx, %%eax                      \n\t"
#endif

//eax:state ebx:low edx:mask esi:range
        "movzbl "MPS_STATE"(%2, %%eax), %%ecx   \n\t"
        "movb %%cl, (%1)                        \n\t"

        "movl %%esi, %%edx                      \n\t"
//eax:bit ebx:low edx:range esi:range

        "shr $19, %%esi                         \n\t"
        "movzbl " MANGLE(ff_h264_norm_shift) "(%%esi), %%ecx   \n\t"
        "shll %%cl, %%ebx                       \n\t"
        "shll %%cl, %%edx                       \n\t"
        "test %%bx, %%bx                        \n\t"
        " jnz 1f                                \n\t"

        "movl "BYTE     "(%2), %%ecx            \n\t"
        "movzwl (%%ecx), %%esi                  \n\t"
        "bswap %%esi                            \n\t"
        "shrl $15, %%esi                        \n\t"
        "subl $0xFFFF, %%esi                    \n\t"
        "addl $2, %%ecx                         \n\t"
        "movl %%ecx, "BYTE    "(%2)             \n\t"

        "leal -1(%%ebx), %%ecx                  \n\t"
        "xorl %%ebx, %%ecx                      \n\t"
        "shrl $17, %%ecx                        \n\t"
        "movzbl " MANGLE(ff_h264_norm_shift) "(%%ecx), %%ecx   \n\t"
        "neg %%cl                               \n\t"
        "add $7, %%cl                           \n\t"

        "shll %%cl , %%esi                      \n\t"
        "addl %%esi, %%ebx                      \n\t"
        "1:                                     \n\t"
        "movl %%edx, "RANGE    "(%2)            \n\t"
        "movl %%ebx, "LOW      "(%2)            \n\t"
        "andl $1, %%eax                         \n\t"
        :"=&a"(bit)
        :"r"(state), "r"(c)
        : "%ecx", "%ebx", "%edx", "%esi"
    );
#endif
#else
    int s = *state;
    int RangeLPS= c->lps_range[s][c->range>>(CABAC_BITS+7)]<<(CABAC_BITS+1);
    int bit, lps_mask attribute_unused;

    c->range -= RangeLPS;
#ifndef BRANCHLESS_CABAD
    if(c->low < c->range){
        bit= s&1;
        *state= c->mps_state[s];
