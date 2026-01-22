            : [outp] "+r" (outp), [inp] "+r" (inp)
            :
            : "q0", "memory");

    while (inp != endp)
        asm volatile (
            "vld1.f32 {q0-q1}, [%[inp]]!\n"
            "vcvt.s32.f32 q0, q0, #28\n"
            "vcvt.s32.f32 q1, q1, #28\n"
            "vst1.s32 {q0-q1}, [%[outp]]!\n"
            : [outp] "+r" (outp), [inp] "+r" (inp)
            :
            : "q0", "q1", "memory");

    outbuf->i_nb_samples = inbuf->i_nb_samples;
    outbuf->i_nb_bytes = inbuf->i_nb_bytes;
    (void) aout;
}

/**
 * Signed 32-bits fixed point to signed 16-bits integer
 */
static void Do_S32_S16 (aout_instance_t *aout, aout_filter_t *filter,
                        aout_buffer_t *inbuf, aout_buffer_t *outbuf)
{
    unsigned nb_samples = inbuf->i_nb_samples
                     * aout_FormatNbChannels (&filter->input);
    int32_t *inp = (int32_t *)inbuf->p_buffer;
    const int32_t *endp = inp + nb_samples;
    int16_t *outp = (int16_t *)outbuf->p_buffer;

    while (nb_samples & 3)
    {
        const int16_t roundup = 1 << 12;
        asm volatile (
            "qadd r0, %[inv], %[roundup]\n"
            "ssat %[outv], #16, r0, asr #13\n"
            : [outv] "=r" (*outp)
            : [inv] "r" (*inp), [roundup] "r" (roundup)
            : "r0");
        inp++;
        outp++;
        nb_samples--;
    }

    if (nb_samples & 4)
