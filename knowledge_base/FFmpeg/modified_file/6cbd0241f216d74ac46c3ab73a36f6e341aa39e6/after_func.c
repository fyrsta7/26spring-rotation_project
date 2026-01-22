    /* If code-block contains no compressed data: nothing to do. */
    if (!cblk->length)
        return 0;

    for (y = 0; y < height+2; y++)
        memset(t1->flags[y], 0, (width + 2)*sizeof(**t1->flags));

    cblk->data[cblk->length] = 0xff;
    cblk->data[cblk->length+1] = 0xff;
    ff_mqc_initdec(&t1->mqc, cblk->data);

    while (passno--) {
        switch(pass_t) {
