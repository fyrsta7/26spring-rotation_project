    AVResampleContext *c= av_mallocz(sizeof(AVResampleContext));
    double factor= FFMIN(out_rate * cutoff / in_rate, 1.0);
    int phase_count= 1<<phase_shift;

    c->phase_shift= phase_shift;
    c->phase_mask= phase_count-1;
    c->linear= linear;

    c->filter_length= FFMAX((int)ceil(filter_size/factor), 1);
    c->filter_bank= av_mallocz(c->filter_length*(phase_count+1)*sizeof(FELEM));
    av_build_filter(c->filter_bank, factor, c->filter_length, phase_count, 1<<FILTER_SHIFT, WINDOW_TYPE);
    memcpy(&c->filter_bank[c->filter_length*phase_count+1], c->filter_bank, (c->filter_length-1)*sizeof(FELEM));
    c->filter_bank[c->filter_length*phase_count]= c->filter_bank[c->filter_length - 1];

    c->src_incr= out_rate;
    c->ideal_dst_incr= c->dst_incr= in_rate * phase_count;
    c->index= -phase_count*((c->filter_length-1)/2);

    return c;
}

void av_resample_close(AVResampleContext *c){
    av_freep(&c->filter_bank);
    av_freep(&c);
}

/**
 * Compensates samplerate/timestamp drift. The compensation is done by changing
 * the resampler parameters, so no audible clicks or similar distortions ocur
 * @param compensation_distance distance in output samples over which the compensation should be performed
 * @param sample_delta number of output samples which should be output less
 *
 * example: av_resample_compensate(c, 10, 500)
 * here instead of 510 samples only 500 samples would be output
 *
 * note, due to rounding the actual compensation might be slightly different,
 * especially if the compensation_distance is large and the in_rate used during init is small
 */
void av_resample_compensate(AVResampleContext *c, int sample_delta, int compensation_distance){
//    sample_delta += (c->ideal_dst_incr - c->dst_incr)*(int64_t)c->compensation_distance / c->ideal_dst_incr;
    c->compensation_distance= compensation_distance;
    c->dst_incr = c->ideal_dst_incr - c->ideal_dst_incr * (int64_t)sample_delta / compensation_distance;
}

/**
 * resamples.
 * @param src an array of unconsumed samples
 * @param consumed the number of samples of src which have been consumed are returned here
 * @param src_size the number of unconsumed samples available
 * @param dst_size the amount of space in samples available in dst
 * @param update_ctx if this is 0 then the context wont be modified, that way several channels can be resampled with the same context
 * @return the number of samples written in dst or -1 if an error occured
 */
int av_resample(AVResampleContext *c, short *dst, short *src, int *consumed, int src_size, int dst_size, int update_ctx){
    int dst_index, i;
    int index= c->index;
    int frac= c->frac;
    int dst_incr_frac= c->dst_incr % c->src_incr;
    int dst_incr=      c->dst_incr / c->src_incr;
    int compensation_distance= c->compensation_distance;

  if(compensation_distance == 0 && c->filter_length == 1 && c->phase_shift==0){
        int64_t index2= ((int64_t)index)<<32;
        int64_t incr= (1LL<<32) * c->dst_incr / c->src_incr;
        dst_size= FFMIN(dst_size, (src_size-1-index) * (int64_t)c->src_incr / c->dst_incr);

        for(dst_index=0; dst_index < dst_size; dst_index++){
            dst[dst_index] = src[index2>>32];
            index2 += incr;
        }
        frac += dst_index * dst_incr_frac;
        index += dst_index * dst_incr;
        index += frac / c->src_incr;
        frac %= c->src_incr;
  }else{
    for(dst_index=0; dst_index < dst_size; dst_index++){
        FELEM *filter= c->filter_bank + c->filter_length*(index & c->phase_mask);
        int sample_index= index >> c->phase_shift;
        FELEM2 val=0;

        if(sample_index < 0){
            for(i=0; i<c->filter_length; i++)
                val += src[FFABS(sample_index + i) % src_size] * filter[i];
        }else if(sample_index + c->filter_length > src_size){
            break;
        }else if(c->linear){
            FELEM2 v2=0;
            for(i=0; i<c->filter_length; i++){
                val += src[sample_index + i] * (FELEM2)filter[i];
