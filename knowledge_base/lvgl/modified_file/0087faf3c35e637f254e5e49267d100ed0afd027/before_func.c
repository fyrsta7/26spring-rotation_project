                else *vdb_buf_tmp = color_mix(color, *vdb_buf_tmp, opa_tmp * px_cnt);
            }

           vdb_buf_tmp++;
        }

        map1_p += font_p->width_byte;
        map2_p += font_p->width_byte;
        map1_p += font_p->width_byte - col_byte_cnt;
        map2_p += font_p->width_byte - col_byte_cnt;
        vdb_buf_tmp += vdb_width  - ((col_end) - (col_start)); /*Next row in VDB*/
    }
#else
    for(row = row_start; row < row_end; row ++) {
        col_byte_cnt = 0;
        col_bit = 7 - (col_start % 8);
        for(col = col_start; col < col_end; col ++) {

            if((*map_p & (1 << col_bit)) != 0) {
                if(opa == OPA_COVER) *vdb_buf_tmp = color;
                else *vdb_buf_tmp = color_mix(color, *vdb_buf_tmp, opa);
            }

           vdb_buf_tmp++;

           if(col_bit != 0) col_bit --;
           else {
               col_bit = 7;
               col_byte_cnt ++;
               map_p ++;
            }
        }

        map_p += font_p->width_byte - col_byte_cnt;
        vdb_buf_tmp += vdb_width  - (col_end - col_start); /*Next row in VDB*/
    }
#endif
}

/**
 * Draw a color map to the display
 * @param cords_p coordinates the color map
 * @param mask_p the map will drawn only on this area  (truncated to VDB area)
 * @param map_p pointer to a color_t array
 * @param opa opacity of the map (ignored, only for compatibility with lv_vmap)
 * @param transp true: enable transparency of LV_IMG_COLOR_TRANSP color pixels
 * @param upscale true: upscale to double size
 * @param recolor mix the pixels with this color
 * @param recolor_opa the intense of recoloring
 */
void lv_vmap(const area_t * cords_p, const area_t * mask_p, 
             const color_t * map_p, opa_t opa, bool transp, bool upscale,
			 color_t recolor, opa_t recolor_opa)
{
    area_t masked_a;
    bool union_ok;
    lv_vdb_t * vdb_p = lv_vdb_get();

    /*Get the union of map size and mask*/
    /* The mask is already truncated to the vdb size
    * in 'lv_refr_area_with_vdb' function */
    union_ok = area_union(&masked_a, cords_p, mask_p);

    /*If there are common part of the three area then draw to the vdb*/
    if(union_ok == false)  return;

    uint8_t ds_shift = 0;
    if(upscale != false) ds_shift = 1;

    /*If the map starts OUT of the masked area then calc. the first pixel*/
    cord_t map_width = area_get_width(cords_p) >> ds_shift;
    if(cords_p->y1 < masked_a.y1) {
        map_p += (uint32_t) map_width * ((masked_a.y1 - cords_p->y1) >> ds_shift);
    }
    if(cords_p->x1 < masked_a.x1) {
        map_p += (masked_a.x1 - cords_p->x1) >> ds_shift;
    }

    /*Stores coordinates relative to the act vdb*/
    masked_a.x1 = masked_a.x1 - vdb_p->area.x1;
    masked_a.y1 = masked_a.y1 - vdb_p->area.y1;
    masked_a.x2 = masked_a.x2 - vdb_p->area.x1;
    masked_a.y2 = masked_a.y2 - vdb_p->area.y1;

    cord_t vdb_width = area_get_width(&vdb_p->area);
    color_t * vdb_buf_tmp = vdb_p->buf;
    vdb_buf_tmp += (uint32_t) vdb_width * masked_a.y1; /*Move to the first row*/

    map_p -= (masked_a.x1 >> ds_shift); /*Move back. It will be easier to index 'map_p' later*/

    /*No upscalse*/
    if(upscale == false) {
        if(transp == false) { /*Simply copy the pixels to the VDB*/
            cord_t row;
            cord_t map_useful_w = area_get_width(&masked_a);

            for(row = masked_a.y1; row <= masked_a.y2; row++) {
#if DISP_HW_ACC == 0
            	sw_color_cpy(&vdb_buf_tmp[masked_a.x1], &map_p[masked_a.x1], map_useful_w, opa);
#else
            	disp_color_cpy(&vdb_buf_tmp[masked_a.x1], &map_p[masked_a.x1], map_useful_w, opa);
#endif
                map_p += map_width;               /*Next row on the map*/
                vdb_buf_tmp += vdb_width;         /*Next row on the VDB*/
            }
            /*To recolor draw simply a rectangle above the image*/
            if(recolor_opa != OPA_TRANSP) {
                lv_vfill(cords_p, mask_p, recolor, recolor_opa);
            }
        } else { /*transp == true: Check all pixels */
            cord_t row;
            cord_t col;
            color_t transp_color = LV_COLOR_TRANSP;

            if(recolor_opa == OPA_TRANSP) {/*No recolor*/
                if(opa == OPA_COVER)  { /*no opa */
                    for(row = masked_a.y1; row <= masked_a.y2; row++) {
                        for(col = masked_a.x1; col <= masked_a.x2; col ++) {
                            if(map_p[col].full != transp_color.full) {
                                vdb_buf_tmp[col] = map_p[col];
                            }
                        }

                        map_p += map_width;         /*Next row on the map*/
                        vdb_buf_tmp += vdb_width;   /*Next row on the VDB*/
                    }
                } else {
                    for(row = masked_a.y1; row <= masked_a.y2; row++) {
                        for(col = masked_a.x1; col <= masked_a.x2; col ++) {
                            if(map_p[col].full != transp_color.full) {
                                vdb_buf_tmp[col] = color_mix( map_p[col], vdb_buf_tmp[col], opa);
                            }
                        }

                        map_p += map_width;          /*Next row on the map*/
                        vdb_buf_tmp += vdb_width;   /*Next row on the VDB*/
                    }
                }
            } else { /*Recolor needed*/
                color_t color_tmp;
                if(opa == OPA_COVER)  { /*no opa */
                    for(row = masked_a.y1; row <= masked_a.y2; row++) {
                        for(col = masked_a.x1; col <= masked_a.x2; col ++) {
                            if(map_p[col].full != transp_color.full) {
                                color_tmp = color_mix(recolor, map_p[col], recolor_opa);
                                vdb_buf_tmp[col] = color_tmp;
                            }
                        }

                        map_p += map_width; /*Next row on the map*/
                        vdb_buf_tmp += vdb_width;         /*Next row on the VDB*/
                    }
                } else {
                    for(row = masked_a.y1; row <= masked_a.y2; row++) {
                        for(col = masked_a.x1; col <= masked_a.x2; col ++) {
                            if(map_p[col].full != transp_color.full) {
                                color_tmp = color_mix(recolor, map_p[col], recolor_opa);
                                vdb_buf_tmp[col] = color_mix(color_tmp, vdb_buf_tmp[col], opa);
                            }
                        }

                        map_p += map_width; /*Next row on the map*/
                        vdb_buf_tmp += vdb_width;         /*Next row on the VDB*/
                    }
                }
            }
        }
    }
    /*Upscalse*/
    else {
        cord_t row;
        cord_t col;
        color_t transp_color = LV_COLOR_TRANSP;
        color_t color_tmp;
        color_t prev_color = COLOR_BLACK;
        cord_t map_col;
