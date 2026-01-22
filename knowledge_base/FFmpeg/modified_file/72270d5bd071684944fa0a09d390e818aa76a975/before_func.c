#include "formats.h"
#include "internal.h"
#include "af_afir.h"

static void fcmul_add_c(float *sum, const float *t, const float *c, ptrdiff_t len)
{
    int n;

    for (n = 0; n < len; n++) {
        const float cre = c[2 * n    ];
        const float cim = c[2 * n + 1];
        const float tre = t[2 * n    ];
        const float tim = t[2 * n + 1];

        sum[2 * n    ] += tre * cre - tim * cim;
        sum[2 * n + 1] += tre * cim + tim * cre;
    }

    sum[2 * n] += t[2 * n] * c[2 * n];
}

static int fir_channel(AVFilterContext *ctx, void *arg, int ch, int nb_jobs)
{
    AudioFIRContext *s = ctx->priv;
    const float *in = (const float *)s->in[0]->extended_data[ch];
    AVFrame *out = arg;
    float *block, *buf, *ptr = (float *)out->extended_data[ch];
    int n, i, j;

    for (int segment = 0; segment < s->nb_segments; segment++) {
        AudioFIRSegment *seg = &s->seg[segment];
        float *src = (float *)seg->input->extended_data[ch];
        float *dst = (float *)seg->output->extended_data[ch];
        float *sum = (float *)seg->sum->extended_data[ch];

        s->fdsp->vector_fmul_scalar(src + seg->input_offset, in, s->dry_gain, FFALIGN(out->nb_samples, 4));
        emms_c();

        seg->output_offset[ch] += s->min_part_size;
        if (seg->output_offset[ch] == seg->part_size) {
            seg->output_offset[ch] = 0;
            memset(dst, 0, sizeof(*dst) * seg->part_size);
        } else {
            memmove(src, src + s->min_part_size, (seg->input_size - s->min_part_size) * sizeof(*src));

            dst += seg->output_offset[ch];
            for (n = 0; n < out->nb_samples; n++) {
                ptr[n] += dst[n];
            }
            continue;
        }

        memset(sum, 0, sizeof(*sum) * seg->fft_length);
        block = (float *)seg->block->extended_data[ch] + seg->part_index[ch] * seg->block_size;
        memset(block + seg->part_size, 0, sizeof(*block) * (seg->fft_length - seg->part_size));

        memcpy(block, src, sizeof(*src) * seg->part_size);

        av_rdft_calc(seg->rdft[ch], block);
        block[2 * seg->part_size] = block[1];
        block[1] = 0;

        j = seg->part_index[ch];

        for (i = 0; i < seg->nb_partitions; i++) {
            const int coffset = j * seg->coeff_size;
            const float *block = (const float *)seg->block->extended_data[ch] + i * seg->block_size;
            const FFTComplex *coeff = (const FFTComplex *)seg->coeff->extended_data[ch * !s->one2many] + coffset;

            s->fcmul_add(sum, block, (const float *)coeff, seg->part_size);

            if (j == 0)
                j = seg->nb_partitions;
            j--;
        }

        sum[1] = sum[2 * seg->part_size];
        av_rdft_calc(seg->irdft[ch], sum);

        buf = (float *)seg->buffer->extended_data[ch];
        for (n = 0; n < seg->part_size; n++) {
            buf[n] += sum[n];
        }

