                                    b2++;
                                }
                            }
                            pv = put_vector(book, pb, vec);
                            if (!pv)
                                return AVERROR(EINVAL);
                            for (dim = book->ndimensions; dim--; ) {
                                coeffs[a1 + b1] -= *pv++;
                                if ((a1 += samples) == s) {
                                    a1 = 0;
                                    b1++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

static int apply_window_and_mdct(vorbis_enc_context *venc,
                                 float **audio, int samples)
{
    int i, channel;
    const float * win = venc->win[0];
    int window_len = 1 << (venc->log2_blocksize[0] - 1);
    float n = (float)(1 << venc->log2_blocksize[0]) / 4.0;
    // FIXME use dsp

    if (!venc->have_saved && !samples)
        return 0;

    if (venc->have_saved) {
        for (channel = 0; channel < venc->channels; channel++)
            memcpy(venc->samples + channel * window_len * 2,
                   venc->saved + channel * window_len, sizeof(float) * window_len);
    } else {
        for (channel = 0; channel < venc->channels; channel++)
            memset(venc->samples + channel * window_len * 2, 0,
                   sizeof(float) * window_len);
    }

    if (samples) {
        for (channel = 0; channel < venc->channels; channel++) {
            float * offset = venc->samples + channel*window_len*2 + window_len;
            for (i = 0; i < samples; i++)
                offset[i] = audio[channel][i] / n * win[window_len - i - 1];
        }
