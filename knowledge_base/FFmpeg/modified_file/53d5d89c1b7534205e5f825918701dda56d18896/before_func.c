{
    int i, csize = 1;
    int32_t *src[3],  i0,  i1,  i2;
    float   *srcf[3], i0f, i1f, i2f;

    for (i = 0; i < 3; i++)
        if (tile->codsty[0].transform == FF_DWT97)
            srcf[i] = tile->comp[i].f_data;
        else
            src [i] = tile->comp[i].i_data;

    for (i = 0; i < 2; i++)
        csize *= tile->comp[0].coord[i][1] - tile->comp[0].coord[i][0];
    switch (tile->codsty[0].transform) {
    case FF_DWT97:
        for (i = 0; i < csize; i++) {
            i0f = *srcf[0] + (f_ict_params[0] * *srcf[2]);
            i1f = *srcf[0] - (f_ict_params[1] * *srcf[1])
                           - (f_ict_params[2] * *srcf[2]);
            i2f = *srcf[0] + (f_ict_params[3] * *srcf[1]);
            *srcf[0]++ = i0f;
            *srcf[1]++ = i1f;
            *srcf[2]++ = i2f;
        }
        break;
    case FF_DWT97_INT:
        for (i = 0; i < csize; i++) {
            i0 = *src[0] + (((i_ict_params[0] * *src[2]) + (1 << 15)) >> 16);
            i1 = *src[0] - (((i_ict_params[1] * *src[1]) + (1 << 15)) >> 16)
                         - (((i_ict_params[2] * *src[2]) + (1 << 15)) >> 16);
            i2 = *src[0] + (((i_ict_params[3] * *src[1]) + (1 << 15)) >> 16);
            *src[0]++ = i0;
            *src[1]++ = i1;
            *src[2]++ = i2;
        }
        break;
    case FF_DWT53:
        for (i = 0; i < csize; i++) {
            i1 = *src[0] - (*src[2] + *src[1] >> 2);
            i0 = i1 + *src[2];
            i2 = i1 + *src[1];
            *src[0]++ = i0;
            *src[1]++ = i1;
            *src[2]++ = i2;
        }
        break;
    }
}

static int jpeg2000_decode_tile(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile,
                                AVFrame *picture)
{
    int compno, reslevelno, bandno;
    int x, y;

    uint8_t *line;
    Jpeg2000T1Context t1;
    /* Loop on tile components */

    for (compno = 0; compno < s->ncomponents; compno++) {
        Jpeg2000Component *comp     = tile->comp + compno;
        Jpeg2000CodingStyle *codsty = tile->codsty + compno;
        /* Loop on resolution levels */
        for (reslevelno = 0; reslevelno < codsty->nreslevels2decode; reslevelno++) {
            Jpeg2000ResLevel *rlevel = comp->reslevel + reslevelno;
            /* Loop on bands */
            for (bandno = 0; bandno < rlevel->nbands; bandno++) {
                uint16_t nb_precincts, precno;
                Jpeg2000Band *band = rlevel->band + bandno;
                int cblkno = 0, bandpos;
                bandpos = bandno + (reslevelno > 0);

                nb_precincts = rlevel->num_precincts_x * rlevel->num_precincts_y;
                /* Loop on precincts */
                for (precno = 0; precno < nb_precincts; precno++) {
                    Jpeg2000Prec *prec = band->prec + precno;

                    /* Loop on codeblocks */
                    for (cblkno = 0; cblkno < prec->nb_codeblocks_width * prec->nb_codeblocks_height; cblkno++) {
                        int x, y;
                        Jpeg2000Cblk *cblk = prec->cblk + cblkno;
                        decode_cblk(s, codsty, &t1, cblk,
                                    cblk->coord[0][1] - cblk->coord[0][0],
                                    cblk->coord[1][1] - cblk->coord[1][0],
                                    bandpos);

                        x = cblk->coord[0][0];
                        y = cblk->coord[1][0];

                        if (codsty->transform == FF_DWT97)
                            dequantization_float(x, y, cblk, comp, &t1, band);
                        else
                            dequantization_int(x, y, cblk, comp, &t1, band);
                   } /* end cblk */
                } /*end prec */
            } /* end band */
        } /* end reslevel */

        /* inverse DWT */
        ff_dwt_decode(&comp->dwt, codsty->transform == FF_DWT97 ? (void*)comp->f_data : (void*)comp->i_data);
    } /*end comp */

    /* inverse MCT transformation */
    if (tile->codsty[0].mct)
        mct_decode(s, tile);

    if (s->precision <= 8) {
        for (compno = 0; compno < s->ncomponents; compno++) {
            Jpeg2000Component *comp = tile->comp + compno;
            float *datap = comp->f_data;
            int32_t *i_datap = comp->i_data;
            y    = tile->comp[compno].coord[1][0] - s->image_offset_y;
            line = picture->data[0] + y * picture->linesize[0];
            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]) {
                uint8_t *dst;

                x   = tile->comp[compno].coord[0][0] - s->image_offset_x;
                dst = line + x * s->ncomponents + compno;

