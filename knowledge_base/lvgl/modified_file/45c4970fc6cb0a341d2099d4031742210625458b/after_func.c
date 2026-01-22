    }
    /*Fill with opacity*/
    else {
        /*Use hw blend if present*/
        if(disp->driver.mem_blend_cb) {
            if(color_array_tmp[0].full != color.full || last_width != w) {
                uint16_t i;
                for(i = 0; i < w; i++) {
                    color_array_tmp[i].full = color.full;
                }

                last_width = w;
            }
            lv_coord_t row;
            for(row = vdb_rel_a.y1; row <= vdb_rel_a.y2; row++) {
                disp->driver.mem_blend_cb(&disp->driver, &vdb_buf_tmp[vdb_rel_a.x1], color_array_tmp, w, opa);
                vdb_buf_tmp += vdb_width;
            }

        }
        /*Use sw fill with opa if no better option*/
        else {
            sw_color_fill(&vdb->area, vdb->buf_act, &vdb_rel_a, color, opa);
        }
    }
#else
    sw_color_fill(&vdb->area, vdb->buf_act, &vdb_rel_a, color, opa);
#endif
}

/**
 * Draw a letter in the Virtual Display Buffer
 * @param pos_p left-top coordinate of the latter
 * @param mask_p the letter will be drawn only on this area  (truncated to VDB area)
 * @param font_p pointer to font
 * @param letter a letter to draw
 * @param color color of letter
 * @param opa opacity of letter (0..255)
 */
void lv_draw_letter(const lv_point_t * pos_p, const lv_area_t * mask_p, const lv_font_t * font_p,
                    uint32_t letter, lv_color_t color, lv_opa_t opa)
{
    const uint8_t bpp1_opa_table[2] = {
        0, 255}; /*Opacity mapping with bpp = 1 (Just for compatibility)*/
    const uint8_t bpp2_opa_table[4]  = {0, 85, 170, 255}; /*Opacity mapping with bpp = 2*/
    const uint8_t bpp4_opa_table[16] = {0,  17, 34,  51,  /*Opacity mapping with bpp = 4*/
                                        68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255};
    if(opa < LV_OPA_MIN) return;
    if(opa > LV_OPA_MAX) opa = LV_OPA_COVER;

    if(font_p == NULL) {
        LV_LOG_WARN("Font: character's bitmap not found");
        return;
    }

    lv_coord_t pos_x = pos_p->x;
    lv_coord_t pos_y = pos_p->y;
    uint8_t letter_w = lv_font_get_real_width(font_p, letter);
    uint8_t letter_h = lv_font_get_height(font_p);
    uint8_t bpp      = lv_font_get_bpp(font_p, letter); /*Bit per pixel (1,2, 4 or 8)*/
    const uint8_t * bpp_opa_table;
    uint8_t mask_init;
    uint8_t mask;

    if(lv_font_is_monospace(font_p, letter)) {
        pos_x += (lv_font_get_width(font_p, letter) - letter_w) / 2;
    }

    switch(bpp) {
        case 1:
            bpp_opa_table = bpp1_opa_table;
            mask_init     = 0x80;
            break;
        case 2:
            bpp_opa_table = bpp2_opa_table;
            mask_init     = 0xC0;
            break;
        case 4:
            bpp_opa_table = bpp4_opa_table;
            mask_init     = 0xF0;
            break;
        case 8:
            bpp_opa_table = NULL;
            mask_init     = 0xFF;
            break;       /*No opa table, pixel value will be used directly*/
        default: return; /*Invalid bpp. Can't render the letter*/
    }

    const uint8_t * map_p = lv_font_get_bitmap(font_p, letter);

    if(map_p == NULL) return;

    /*If the letter is completely out of mask don't draw it */
    if(pos_x + letter_w < mask_p->x1 || pos_x > mask_p->x2 || pos_y + letter_h < mask_p->y1 ||
       pos_y > mask_p->y2)
        return;

    lv_disp_t * disp    = lv_refr_get_disp_refreshing();
    lv_disp_buf_t * vdb = lv_disp_get_buf(disp);

    lv_coord_t vdb_width     = lv_area_get_width(&vdb->area);
    lv_color_t * vdb_buf_tmp = vdb->buf_act;
    lv_coord_t col, row;
    uint8_t col_bit;
    uint8_t col_byte_cnt;

    /*Width in bytes (on the screen finally) (e.g. w = 11 -> 2 bytes wide)*/
    uint8_t width_byte_scr = letter_w >> 3;
    if(letter_w & 0x7) width_byte_scr++;

    /*Letter width in byte. Real width in the font*/
    uint8_t width_byte_bpp = (letter_w * bpp) >> 3;
    if((letter_w * bpp) & 0x7) width_byte_bpp++;

    /* Calculate the col/row start/end on the map*/
    lv_coord_t col_start = pos_x >= mask_p->x1 ? 0 : mask_p->x1 - pos_x;
    lv_coord_t col_end   = pos_x + letter_w <= mask_p->x2 ? letter_w : mask_p->x2 - pos_x + 1;
    lv_coord_t row_start = pos_y >= mask_p->y1 ? 0 : mask_p->y1 - pos_y;
    lv_coord_t row_end   = pos_y + letter_h <= mask_p->y2 ? letter_h : mask_p->y2 - pos_y + 1;

    /*Set a pointer on VDB to the first pixel of the letter*/
    vdb_buf_tmp += ((pos_y - vdb->area.y1) * vdb_width) + pos_x - vdb->area.x1;

    /*If the letter is partially out of mask the move there on VDB*/
    vdb_buf_tmp += (row_start * vdb_width) + col_start;

    /*Move on the map too*/
    map_p += (row_start * width_byte_bpp) + ((col_start * bpp) >> 3);

    uint8_t letter_px;
    lv_opa_t px_opa;
    for(row = row_start; row < row_end; row++) {
        col_byte_cnt = 0;
        col_bit      = (col_start * bpp) % 8;
        mask         = mask_init >> col_bit;
        for(col = col_start; col < col_end; col++) {
            letter_px = (*map_p & mask) >> (8 - col_bit - bpp);
            if(letter_px != 0) {
                if(opa == LV_OPA_COVER) {
                    px_opa = bpp == 8 ? letter_px : bpp_opa_table[letter_px];
