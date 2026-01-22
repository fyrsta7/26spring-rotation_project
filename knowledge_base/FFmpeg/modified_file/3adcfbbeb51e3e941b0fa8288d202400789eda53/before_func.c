                res_change++;

            if((code&0x300)==0x200 && src_fmt){
                valid_psc++;
            }else
                invalid_psc++;
            last_src_fmt= src_fmt;
        }
    }
//av_log(NULL, AV_LOG_ERROR, "h263_probe: psc:%d invalid:%d res_change:%d\n", valid_psc, invalid_psc, res_change);
//h263_probe: psc:3 invalid:0 res_change:0 (1588/recent_ffmpeg_parses_mpg_incorrectly.mpg)
    if(valid_psc > 2*invalid_psc + 2*res_change + 3){
        return 50;
    }else if(valid_psc > 2*invalid_psc)
        return 25;
    return 0;
}
#endif

#if CONFIG_H261_DEMUXER
static int h261_probe(AVProbeData *p)
{
    uint32_t code= -1;
    int i;
    int valid_psc=0;
    int invalid_psc=0;
    int next_gn=0;
    int src_fmt=0;
    GetBitContext gb;

    init_get_bits(&gb, p->buf, p->buf_size*8);

    for(i=0; i<p->buf_size*8; i++){
        code = (code<<1) + get_bits1(&gb);
