#define XING_FLAG_FRAMES 0x01
#define XING_FLAG_SIZE   0x02
#define XING_FLAG_TOC    0x04

#define XING_TOC_COUNT 100

typedef struct {
    AVClass *class;
    int64_t filesize;
    int64_t header_filesize;
    int xing_toc;
    int start_pad;
    int end_pad;
    int usetoc;
    int is_cbr;
} MP3DecContext;

/* mp3 read */

static int mp3_read_probe(AVProbeData *p)
{
    int max_frames, first_frames = 0;
    int fsize, frames, sample_rate;
    uint32_t header;
    const uint8_t *buf, *buf0, *buf2, *end;
    AVCodecContext avctx;

    buf0 = p->buf;
    end = p->buf + p->buf_size - sizeof(uint32_t);
    while(buf0 < end && !*buf0)
        buf0++;

    max_frames = 0;
    buf = buf0;

    for(; buf < end; buf= buf2+1) {
        buf2 = buf;
        if(ff_mpa_check_header(AV_RB32(buf2)))
            continue;

        for(frames = 0; buf2 < end; frames++) {
            header = AV_RB32(buf2);
            fsize = avpriv_mpa_decode_header(&avctx, header, &sample_rate, &sample_rate, &sample_rate, &sample_rate);
