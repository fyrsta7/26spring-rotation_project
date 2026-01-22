                count += getbit(&gb, &bitbuf, &bits);
                offset = bytestream2_get_byte(&gb) - 0x0100;
            }
            count  += 2;
            offset += writeoffset;
            if (offset < 0 || offset + count >= hnm->width * hnm->height) {
                av_log(avctx, AV_LOG_ERROR, "Attempting to read out of bounds\n");
                break;
            } else if (writeoffset + count >= hnm->width * hnm->height) {
                av_log(avctx, AV_LOG_ERROR,
                       "Attempting to write out of bounds\n");
                break;
            }
            while (count--) {
                hnm->current[writeoffset++] = hnm->current[offset++];
            }
        }
