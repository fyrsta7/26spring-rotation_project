    /* rotate and multiply */
    c = (b = (a = n + i) + j) - i;
    fp = st1 + i;
    for (x=0; x < b; x++) {
        if (x == c)
            fp=in;
        work[x] = *(table++) * (*(st1++) = *(fp++));
    }

    prodsum(buffer1, work + n, i, n);
    prodsum(buffer2, work + a, j, n);

    for (x=0;x<=n;x++) {
        *st2 = *st2 * (0.5625) + buffer1[x];
        out[x] = *(st2++) + buffer2[x];
    }
    *out *= 1.00390625; /* to prevent clipping */
}

static void update(Real288_internal *glob)
{
    int x,y;
    float buffer1[40], temp1[37];
    float buffer2[8], temp2[11];
