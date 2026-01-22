    FlacFrame *frame;

    frame = &s->frame;
    for(i=0,j=0; i<frame->blocksize; i++) {
        for(ch=0; ch<s->channels; ch++,j++) {
            frame->subframes[ch].samples[i] = samples[j];
        }
    }
}


#define rice_encode_count(sum, n, k) (((n)*((k)+1))+((sum-(n>>1))>>(k)))

/**
 * Solve for d/dk(rice_encode_count) = n-((sum-(n>>1))>>(k+1)) = 0
 */
static int find_optimal_param(uint32_t sum, int n)
{
    int k;
    uint32_t sum2;

    if(sum <= n>>1)
