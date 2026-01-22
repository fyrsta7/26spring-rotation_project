    unsigned char ival = byte + 0x16;
    const unsigned char * ptr = src + byte*2;
    unsigned char val = ival;
    int counter = 0;
    unsigned char *dest_end = dest + dest_len;

    unsigned char bits = *ptr++;

    while ( val != 0x16 ) {
        if ( (1 << counter) & bits )
            val = src[byte + val - 0x17];
        else
            val = src[val - 0x17];

        if ( val < 0x16 ) {
            if (dest + 1 > dest_end)
                return 0;
            *dest++ = val;
            val = ival;
        }

        if (counter++ == 7) {
            counter = 0;
            bits = *ptr++;
        }
    }

    return 0;
}

static void xan_unpack(unsigned char *dest, const unsigned char *src, int dest_len)
{
    unsigned char opcode;
    int size;
    int offset;
    int byte1, byte2, byte3;
    unsigned char *dest_end = dest + dest_len;

    for (;;) {
        opcode = *src++;

        if ( (opcode & 0x80) == 0 ) {

            offset = *src++;

            size = opcode & 3;
            if (dest + size > dest_end)
                return;
            memcpy(dest, src, size);  dest += size;  src += size;

            size = ((opcode & 0x1c) >> 2) + 3;
            if (dest + size > dest_end)
                return;
            bytecopy (dest, dest - (((opcode & 0x60) << 3) + offset + 1), size);
            dest += size;

        } else if ( (opcode & 0x40) == 0 ) {

            byte1 = *src++;
            byte2 = *src++;

            size = byte1 >> 6;
            if (dest + size > dest_end)
                return;
            memcpy(dest, src, size);  dest += size;  src += size;

            size = (opcode & 0x3f) + 4;
            if (dest + size > dest_end)
                return;
            bytecopy (dest, dest - (((byte1 & 0x3f) << 8) + byte2 + 1), size);
            dest += size;

        } else if ( (opcode & 0x20) == 0 ) {

            byte1 = *src++;
