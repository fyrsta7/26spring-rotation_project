
    return n;
}

static inline int av_log2_16bit(unsigned int v)
{
    int n;

    n = 0;
    if (v & 0xff00) {
        v >>= 8;
        n += 8;
    }
    n += ff_log2_tab[v];

    return n;
}

/* median of 3 */
static inline int mid_pred(int a, int b, int c)
{
#if (defined(ARCH_X86) && __CPU__ >= 686 || defined(ARCH_X86_64)) && !defined(RUNTIME_CPUDETECT)
    int i=a, j=a;
    asm volatile(
        "cmp    %4, %2 \n\t"
        "cmovg  %4, %0 \n\t"
        "cmovl  %4, %1 \n\t"
        "cmp    %4, %3 \n\t"
        "cmovg  %3, %0 \n\t"
        "cmovl  %3, %1 \n\t"
        "cmp    %3, %2 \n\t"
        "cmovl  %1, %0 \n\t"
        :"+&r"(i), "+&r"(j)
        :"r"(a), "r"(b), "r"(c)
    );
    return i;
#elif 0
    int t= (a-b)&((a-b)>>31);
    a-=t;
    b+=t;
