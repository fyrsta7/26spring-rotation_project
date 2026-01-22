            default: /* 6 to 15 */
                mantissa = get_bits(gbc, quantization_tab[bap]);
                /* Shift mantissa and sign-extend it. */
                mantissa = (mantissa << (32-quantization_tab[bap]))>>8;
                break;
        }
        coeffs[freq] = mantissa >> exps[freq];
    }
}

/**
 * Remove random dithering from coupling range coefficients with zero-bit
 * mantissas for coupled channels which do not use dithering.
 * reference: Section 7.3.4 Dither for Zero Bit Mantissas (bap=0)
 */
static void remove_dithering(AC3DecodeContext *s) {
    int ch, i;

    for(ch=1; ch<=s->fbw_channels; ch++) {
