                lv_draw_img_dsc_t dsc;
                lv_draw_img_dsc_init(&dsc);
                dsc.opa = disp_refr->bg_opa;
                lv_draw_img(draw_ctx, &dsc, &a, disp_refr->bg_img);
            }
            else {
                LV_LOG_WARN("Can't draw the background image");
            }
        }
        else {
            lv_draw_rect_dsc_t dsc;
            lv_draw_rect_dsc_init(&dsc);
            dsc.bg_color = disp_refr->bg_color;
            dsc.bg_opa = disp_refr->bg_opa;
            lv_draw_rect(draw_ctx, &dsc, draw_ctx->buf_area);
        }
    }

    if(disp_refr->draw_prev_over_act) {
        if(top_act_scr == NULL) top_act_scr = disp_refr->act_scr;
        refr_obj_and_children(draw_ctx, top_act_scr);

        /*Refresh the previous screen if any*/
        if(disp_refr->prev_scr) {
            if(top_prev_scr == NULL) top_prev_scr = disp_refr->prev_scr;
            refr_obj_and_children(draw_ctx, top_prev_scr);
        }
    }
    else {
        /*Refresh the previous screen if any*/
        if(disp_refr->prev_scr) {
            if(top_prev_scr == NULL) top_prev_scr = disp_refr->prev_scr;
            refr_obj_and_children(draw_ctx, top_prev_scr);
        }
