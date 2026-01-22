
    /*Normal refresh: draw the area in parts*/
    /*Calculate the max row num*/
    lv_coord_t w = lv_area_get_width(area_p);
    lv_coord_t h = lv_area_get_height(area_p);
    lv_coord_t y2 = area_p->y2 >= lv_disp_get_ver_res(disp_refr) ?
                    lv_disp_get_ver_res(disp_refr) - 1 : area_p->y2;

    int32_t max_row = get_max_row(disp_refr, w, h);

    lv_coord_t row;
    lv_coord_t row_last = 0;
    lv_area_t sub_area;
    for(row = area_p->y1; row + max_row - 1 <= y2; row += max_row) {
        /*Calc. the next y coordinates of draw_buf*/
        sub_area.x1 = area_p->x1;
        sub_area.x2 = area_p->x2;
        sub_area.y1 = row;
        sub_area.y2 = row + max_row - 1;
        layer->buf_area = sub_area;
        layer->clip_area = sub_area;
        layer->buf = disp_refr->draw_buf_act;
        if(sub_area.y2 > y2) sub_area.y2 = y2;
        row_last = sub_area.y2;
        if(y2 == row_last) disp_refr->last_part = 1;
        refr_area_part(layer);
    }

    /*If the last y coordinates are not handled yet ...*/
    if(y2 != row_last) {
        /*Calc. the next y coordinates of draw_buf*/
        sub_area.x1 = area_p->x1;
        sub_area.x2 = area_p->x2;
        sub_area.y1 = row;
        sub_area.y2 = y2;
        layer->buf_area = sub_area;
        layer->clip_area = sub_area;
        layer->buf = disp_refr->draw_buf_act;
        disp_refr->last_part = 1;
        refr_area_part(layer);
    }
}

static void refr_area_part(lv_layer_t * layer)
{
    disp_refr->refreshed_area = layer->clip_area;

    /* In single buffered mode wait here until the buffer is freed.
     * Else we would draw into the buffer while it's still being transferred to the display*/
    if(!lv_disp_is_double_buffered(disp_refr)) {
        while(disp_refr->flushing);
    }
    /*If the screen is transparent initialize it when the flushing is ready*/
    if(lv_color_format_has_alpha(disp_refr->color_format)) {
