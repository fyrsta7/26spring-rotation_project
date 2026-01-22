static void fill_set_px(const lv_area_t * disp_area, lv_color_t * disp_buf,  const lv_area_t * draw_area,
        lv_color_t color, lv_opa_t opa,
        const lv_opa_t * mask, lv_draw_mask_res_t mask_res)
{

    lv_disp_t * disp = lv_refr_get_disp_refreshing();

    /*Get the width of the `disp_area` it will be used to go to the next line*/
    int32_t disp_w = lv_area_get_width(disp_area);

    int32_t x;
    int32_t y;

    if(mask_res == LV_DRAW_MASK_RES_FULL_COVER) {
        for(y = draw_area->y1; y <= draw_area->y2; y++) {
            for(x = draw_area->x1; x <= draw_area->x2; x++) {
                disp->driver.set_px_cb(&disp->driver, (void*)disp_buf, disp_w, x, y, color, opa);
            }
        }
    } else {
        /* The mask is relative to the clipped area.
         * In the cycles below mask will be indexed from `draw_area.x1`
         * but it corresponds to zero index. So prepare `mask_tmp` accordingly. */
        const lv_opa_t * mask_tmp = mask - draw_area->x1;

        /*Get the width of the `draw_area` it will be used to go to the next line of the mask*/
        int32_t draw_area_w = lv_area_get_width(draw_area);

        for(y = draw_area->y1; y <= draw_area->y2; y++) {
            for(x = draw_area->x1; x <= draw_area->x2; x++) {
                disp->driver.set_px_cb(&disp->driver, (void*)disp_buf, disp_w, x, y, color, (uint32_t)((uint32_t)opa * mask_tmp[x]) >> 8);
            }
            mask_tmp += draw_area_w;
        }
    }
}

static void fill_normal(const lv_area_t * disp_area, lv_color_t * disp_buf,  const lv_area_t * draw_area,
        lv_color_t color, lv_opa_t opa,
        const lv_opa_t * mask, lv_draw_mask_res_t mask_res)
{

#if LV_USE_GPU
    lv_disp_t * disp = lv_refr_get_disp_refreshing();
#endif

    /*Get the width of the `disp_area` it will be used to go to the next line*/
    int32_t disp_w = lv_area_get_width(disp_area);

    /*Get the width of the `draw_area` it will be used to go to the next line of the mask*/
    int32_t draw_area_w = lv_area_get_width(draw_area);

    /*Create a temp. disp_buf which always point to current line to draw*/
    lv_color_t * disp_buf_tmp = disp_buf + disp_w * draw_area->y1;

    int32_t x;
    int32_t y;

    /*Simple fill (maybe with opacity), no masking*/
    if(mask_res == LV_DRAW_MASK_RES_FULL_COVER) {
        if(opa > LV_OPA_MAX) {
            lv_color_t * disp_buf_tmp_ori =  disp_buf_tmp;

#if LV_USE_GPU
            if(disp->driver.gpu_fill_cb && draw_area_w > GPU_WIDTH_LIMIT) {
                disp->driver.gpu_fill_cb(&disp->driver, disp_buf, disp_w, draw_area, color);
                return;
            }
#endif

            /*Fill the first line. Use `memcpy` because it's faster then simple value assignment*/
            /*Set the first pixels manually*/
            int32_t direct_fill_end = LV_MATH_MIN(draw_area->x2, draw_area->x1 + FILL_DIRECT_LEN + (draw_area_w & FILL_DIRECT_MASK) - 1);
            for(x = draw_area->x1; x <= direct_fill_end ; x++) {
                disp_buf_tmp[x].full = color.full;
            }

            for(; x <= draw_area->x2; x += FILL_DIRECT_LEN) {
                memcpy(&disp_buf_tmp[x], &disp_buf_tmp[draw_area->x1], FILL_DIRECT_LEN * sizeof(lv_color_t));
            }

            disp_buf_tmp += disp_w;

            for(y = draw_area->y1 + 1; y <= draw_area->y2; y++) {
                memcpy(&disp_buf_tmp[draw_area->x1], &disp_buf_tmp_ori[draw_area->x1], draw_area_w * sizeof(lv_color_t));
                disp_buf_tmp += disp_w;
            }
        }
        else {

#if LV_USE_GPU
            if(disp->driver.gpu_blend_cb && draw_area_w > GPU_WIDTH_LIMIT) {
                static lv_color_t blend_buf[LV_HOR_RES_MAX];
                for(x = 0; x < draw_area_w ; x++) blend_buf[x].full = color.full;

                for(y = draw_area->y1; y <= draw_area->y2; y++) {
                    disp->driver.gpu_blend_cb(&disp->driver, &disp_buf_tmp[draw_area->x1], blend_buf, draw_area_w, opa);
                    disp_buf_tmp += disp_w;
                }
                return;
            }
#endif
            lv_color_t last_dest_color = LV_COLOR_BLACK;
            lv_color_t last_res_color = lv_color_mix(color, last_dest_color, opa);
            for(y = draw_area->y1; y <= draw_area->y2; y++) {
                for(x = draw_area->x1; x <= draw_area->x2; x++) {
                    if(last_dest_color.full != disp_buf_tmp[x].full) {
                        last_dest_color = disp_buf_tmp[x];

#if LV_COLOR_SCREEN_TRANSP
                        if(disp->driver.screen_transp) {
                            lv_color_mix_with_alpha(disp_buf_tmp[x], disp_buf_tmp[x].ch.alpha, color, opa, &last_res_color, &last_res_color.ch.alpha);
                        } else
#endif
                        {
                            last_res_color = lv_color_mix(color, disp_buf_tmp[x], opa);
                        }
                    }
                    disp_buf_tmp[x] = last_res_color;
                }
                disp_buf_tmp += disp_w;
            }
        }
    }
    /*Masked*/
    else {
        /* The mask is relative to the clipped area.
         * In the cycles below mask will be indexed from `draw_area.x1`
         * but it corresponds to zero index. So prepare `mask_tmp` accordingly. */
        const lv_opa_t * mask_tmp = mask - draw_area->x1;

        /*Buffer the result color to avoid recalculating the same color*/
        lv_color_t last_dest_color;
        lv_color_t last_res_color;
        lv_opa_t last_mask = LV_OPA_TRANSP;
        last_dest_color.full = disp_buf_tmp[0].full;
        last_res_color.full = disp_buf_tmp[0].full;

        /*Only the mask matters*/
        if(opa > LV_OPA_MAX) {
            for(y = draw_area->y1; y <= draw_area->y2; y++) {
                for(x = draw_area->x1; x <= draw_area->x2; x++) {
                    if(mask_tmp[x] == 0) continue;
                    if(mask_tmp[x] != last_mask || last_dest_color.full != disp_buf_tmp[x].full)
                    {
#if LV_COLOR_SCREEN_TRANSP
                        if(disp->driver.screen_transp) {
                            lv_color_mix_with_alpha(disp_buf_tmp[x], disp_buf_tmp[x].ch.alpha, color, mask_tmp[x], &last_res_color, &last_res_color.ch.alpha);
                        } else
#endif
                        {
                            if(mask_tmp[x] > LV_OPA_MAX) last_res_color = color;
                            else if(mask_tmp[x] < LV_OPA_MIN) last_res_color = disp_buf_tmp[x];
                            else if(disp_buf_tmp[x].full == color.full) last_res_color = color;
