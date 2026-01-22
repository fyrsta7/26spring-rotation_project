    x4->out_pic.key_frame = pic_out.i_type == X264_TYPE_IDR;
    x4->out_pic.quality = (pic_out.i_qpplus1 - 1) * FF_QP2LAMBDA;

    return bufsize;
}

static int
X264_close(AVCodecContext *avctx)
{
    X264Context *x4 = avctx->priv_data;

    if(x4->enc)
	x264_encoder_close(x4->enc);

    return 0;
}

extern int
X264_init(AVCodecContext *avctx)
{
    X264Context *x4 = avctx->priv_data;

    x264_param_default(&x4->params);

    x4->params.pf_log = X264_log;
    x4->params.p_log_private = avctx;

    x4->params.i_keyint_max = avctx->gop_size;
    x4->params.rc.i_bitrate = avctx->bit_rate / 1000;
    x4->params.rc.i_vbv_buffer_size = avctx->rc_buffer_size / 1000;
    if(avctx->rc_buffer_size)
        x4->params.rc.b_cbr = 1;
    x4->params.i_bframe = avctx->max_b_frames;
    x4->params.b_cabac = avctx->coder_type == FF_CODER_TYPE_AC;

    x4->params.rc.i_qp_min = avctx->qmin;
    x4->params.rc.i_qp_max = avctx->qmax;
    x4->params.rc.i_qp_step = avctx->max_qdiff;

    if(avctx->flags & CODEC_FLAG_QSCALE && avctx->global_quality > 0)
        x4->params.rc.i_qp_constant =
