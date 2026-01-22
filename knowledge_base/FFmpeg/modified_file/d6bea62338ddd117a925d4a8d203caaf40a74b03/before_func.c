        case 1:
            s->frame_size += padding  * 4;
            s->bit_rate = (s->frame_size * sample_rate) / 48000;
            break;
        case 2:
            s->frame_size += padding;
            s->bit_rate = (s->frame_size * sample_rate) / 144000;
            break;
        default:
        case 3:
            s->frame_size += padding;
            s->bit_rate = (s->frame_size * (sample_rate << s->lsf)) / 144000;
            break;
        }
    }

#if defined(DEBUG)
    dprintf("layer%d, %d Hz, %d kbits/s, ",
           s->layer, s->sample_rate, s->bit_rate);
    if (s->nb_channels == 2) {
        if (s->layer == 3) {
            if (s->mode_ext & MODE_EXT_MS_STEREO)
                dprintf("ms-");
            if (s->mode_ext & MODE_EXT_I_STEREO)
                dprintf("i-");
        }
        dprintf("stereo");
    } else {
        dprintf("mono");
    }
    dprintf("\n");
#endif
    return 0;
}
