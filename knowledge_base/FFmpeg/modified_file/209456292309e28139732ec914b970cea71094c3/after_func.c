        if (s->weight_func)
            s->weight_func(s->mcscratch, p->stride, s->weight_log2denom,
                           s->weight[0] + s->weight[1], p->yblen);
        break;
    case 3:
        idx = mc_subpel(s, block, src, dstx, dsty, 0, plane);
        s->put_pixels_tab[idx](s->mcscratch, src, p->stride, p->yblen);
        idx = mc_subpel(s, block, src, dstx, dsty, 1, plane);
        if (s->biweight_func) {
            /* fixme: +32 is a quick hack */
            s->put_pixels_tab[idx](s->mcscratch + 32, src, p->stride, p->yblen);
            s->biweight_func(s->mcscratch, s->mcscratch+32, p->stride, s->weight_log2denom,
                             s->weight[0], s->weight[1], p->yblen);
        } else
            s->avg_pixels_tab[idx](s->mcscratch, src, p->stride, p->yblen);
        break;
    }
    s->add_obmc(mctmp, s->mcscratch, p->stride, obmc_weight, p->yblen);
}

static void mc_row(DiracContext *s, DiracBlock *block, uint16_t *mctmp, int plane, int dsty)
{
    Plane *p = &s->plane[plane];
    int x, dstx = p->xbsep - p->xoffset;

    block_mc(s, block, mctmp, s->obmc_weight[0], plane, -p->xoffset, dsty);
    mctmp += p->xbsep;

    for (x = 1; x < s->blwidth-1; x++) {
        block_mc(s, block+x, mctmp, s->obmc_weight[1], plane, dstx, dsty);
        dstx  += p->xbsep;
        mctmp += p->xbsep;
    }
    block_mc(s, block+x, mctmp, s->obmc_weight[2], plane, dstx, dsty);
}

static void select_dsp_funcs(DiracContext *s, int width, int height, int xblen, int yblen)
{
    int idx = 0;
    if (xblen > 8)
        idx = 1;
    if (xblen > 16)
        idx = 2;

    memcpy(s->put_pixels_tab, s->diracdsp.put_dirac_pixels_tab[idx], sizeof(s->put_pixels_tab));
    memcpy(s->avg_pixels_tab, s->diracdsp.avg_dirac_pixels_tab[idx], sizeof(s->avg_pixels_tab));
    s->add_obmc = s->diracdsp.add_dirac_obmc[idx];
    if (s->weight_log2denom > 1 || s->weight[0] != 1 || s->weight[1] != 1) {
        s->weight_func   = s->diracdsp.weight_dirac_pixels_tab[idx];
        s->biweight_func = s->diracdsp.biweight_dirac_pixels_tab[idx];
    } else {
        s->weight_func   = NULL;
        s->biweight_func = NULL;
    }
}

static int interpolate_refplane(DiracContext *s, DiracFrame *ref, int plane, int width, int height)
{
    /* chroma allocates an edge of 8 when subsampled
       which for 4:2:2 means an h edge of 16 and v edge of 8
       just use 8 for everything for the moment */
    int i, edge = EDGE_WIDTH/2;

    ref->hpel[plane][0] = ref->avframe->data[plane];
    s->mpvencdsp.draw_edges(ref->hpel[plane][0], ref->avframe->linesize[plane], width, height, edge, edge, EDGE_TOP | EDGE_BOTTOM); /* EDGE_TOP | EDGE_BOTTOM values just copied to make it build, this needs to be ensured */

    /* no need for hpel if we only have fpel vectors */
    if (!s->mv_precision)
        return 0;

    for (i = 1; i < 4; i++) {
        if (!ref->hpel_base[plane][i])
            ref->hpel_base[plane][i] = av_malloc((height+2*edge) * ref->avframe->linesize[plane] + 32);
        if (!ref->hpel_base[plane][i]) {
            return AVERROR(ENOMEM);
        }
        /* we need to be 16-byte aligned even for chroma */
        ref->hpel[plane][i] = ref->hpel_base[plane][i] + edge*ref->avframe->linesize[plane] + 16;
    }

    if (!ref->interpolated[plane]) {
        s->diracdsp.dirac_hpel_filter(ref->hpel[plane][1], ref->hpel[plane][2],
                                      ref->hpel[plane][3], ref->hpel[plane][0],
                                      ref->avframe->linesize[plane], width, height);
        s->mpvencdsp.draw_edges(ref->hpel[plane][1], ref->avframe->linesize[plane], width, height, edge, edge, EDGE_TOP | EDGE_BOTTOM);
        s->mpvencdsp.draw_edges(ref->hpel[plane][2], ref->avframe->linesize[plane], width, height, edge, edge, EDGE_TOP | EDGE_BOTTOM);
        s->mpvencdsp.draw_edges(ref->hpel[plane][3], ref->avframe->linesize[plane], width, height, edge, edge, EDGE_TOP | EDGE_BOTTOM);
    }
    ref->interpolated[plane] = 1;

    return 0;
}

