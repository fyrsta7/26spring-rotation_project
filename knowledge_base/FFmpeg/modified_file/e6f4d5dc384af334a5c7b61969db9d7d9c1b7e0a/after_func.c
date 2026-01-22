 *
 * Copyright (c) 2003-2016 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

static av_always_inline int RENAME(decode_line)(FFV1Context *s, int w,
                                                 TYPE *sample[2],
                                                 int plane_index, int bits)
{
    PlaneContext *const p = &s->plane[plane_index];
    RangeCoder *const c   = &s->c;
    int x;
    int run_count = 0;
    int run_mode  = 0;
    int run_index = s->run_index;

    if (is_input_end(s))
        return AVERROR_INVALIDDATA;

    if (s->slice_coding_mode == 1) {
        int i;
        for (x = 0; x < w; x++) {
            int v = 0;
            for (i=0; i<bits; i++) {
                uint8_t state = 128;
                v += v + get_rac(c, &state);
            }
            sample[1][x] = v;
        }
        return 0;
    }

    for (x = 0; x < w; x++) {
        int diff, context, sign;

        if (!(x & 1023)) {
            if (is_input_end(s))
                return AVERROR_INVALIDDATA;
        }

        context = RENAME(get_context)(p, sample[1] + x, sample[0] + x, sample[1] + x);
        if (context < 0) {
            context = -context;
            sign    = 1;
        } else
            sign = 0;

        av_assert2(context < p->context_count);

        if (s->ac != AC_GOLOMB_RICE) {
            diff = get_symbol_inline(c, p->state[context], 1);
        } else {
            if (context == 0 && run_mode == 0)
                run_mode = 1;

            if (run_mode) {
                if (run_count == 0 && run_mode == 1) {
                    if (get_bits1(&s->gb)) {
                        run_count = 1 << ff_log2_run[run_index];
                        if (x + run_count <= w)
                            run_index++;
                    } else {
                        if (ff_log2_run[run_index])
                            run_count = get_bits(&s->gb, ff_log2_run[run_index]);
                        else
                            run_count = 0;
                        if (run_index)
                            run_index--;
                        run_mode = 2;
                    }
                }
                if (sample[1][x - 1] == sample[0][x - 1]) {
                    while (run_count > 1 && w-x > 1) {
                        sample[1][x] = sample[0][x];
                        x++;
                        run_count--;
                    }
                } else {
                while (run_count > 1 && w-x > 1) {
                    sample[1][x] = RENAME(predict)(sample[1] + x, sample[0] + x);
                    x++;
                    run_count--;
                }
                }
                run_count--;
                if (run_count < 0) {
                    run_mode  = 0;
                    run_count = 0;
                    diff      = get_vlc_symbol(&s->gb, &p->vlc_state[context],
