
#define av_clip_int8 av_clip_int8_arm
static av_always_inline av_const int av_clip_int8_arm(int a)
{
    int x;
    __asm__ ("ssat %0, #8,  %1" : "=r"(x) : "r"(a));
