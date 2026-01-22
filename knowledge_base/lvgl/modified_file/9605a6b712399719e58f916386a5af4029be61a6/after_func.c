                    + pos_p->x - vdb_p->area.x1;

    /*If the letter is partially out of mask the move there on VDB*/
    vdb_buf_tmp += (row_start * vdb_width) + col_start;

    /*Move on the map too*/
    map_p += (row_start * width_byte_bpp) + ((col_start * bpp) >> 3);

    uint8_t letter_px;
    for(row = row_start; row < row_end; row ++) {
        col_byte_cnt = 0;
        col_bit = (col_start * bpp) % 8;
        mask = mask_init >> col_bit;
        for(col = col_start; col < col_end; col ++) {
            letter_px = (*map_p & mask) >> (8 - col_bit - bpp);
            if(letter_px != 0) {
                *vdb_buf_tmp = lv_color_mix(color, *vdb_buf_tmp, bpp == 8 ? letter_px : bpp_opa_table[letter_px]);
            }

            vdb_buf_tmp++;

            if(col_bit < 8 - bpp) {
                col_bit += bpp;
                mask = mask >> bpp;
            }
            else {
                col_bit = 0;
                col_byte_cnt ++;
                mask = mask_init;
                map_p ++;
            }
        }

        map_p += (width_byte_bpp) - col_byte_cnt;
        vdb_buf_tmp += vdb_width  - (col_end - col_start); /*Next row in VDB*/
    }
}

/**
 * Draw a color map to the display (image)
 * @param cords_p coordinates the color map
 * @param mask_p the map will drawn only on this area  (truncated to VDB area)
 * @param map_p pointer to a lv_color_t array
 * @param opa opacity of the map
 * @param chroma_keyed true: enable transparency of LV_IMG_LV_COLOR_TRANSP color pixels
 * @param alpha_byte true: extra alpha byte is inserted for every pixel
 * @param recolor mix the pixels with this color
 * @param recolor_opa the intense of recoloring
 */
void lv_vmap(const lv_area_t * cords_p, const lv_area_t * mask_p, 
             const uint8_t * map_p, lv_opa_t opa, bool chroma_key, bool alpha_byte,
			 lv_color_t recolor, lv_opa_t recolor_opa)
{
    lv_area_t masked_a;
    bool union_ok;
    lv_vdb_t * vdb_p = lv_vdb_get();

    /*Get the union of map size and mask*/
    /* The mask is already truncated to the vdb size
    * in 'lv_refr_area_with_vdb' function */
    union_ok = lv_area_union(&masked_a, cords_p, mask_p);

    /*If there are common part of the three area then draw to the vdb*/
    if(union_ok == false)  return;

    /*The pixel size in byte is different if an alpha byte is added too*/
    uint8_t px_size_byte = alpha_byte ? LV_IMG_PX_SIZE_ALPHA_BYTE : sizeof(lv_color_t);

    /*If the map starts OUT of the masked area then calc. the first pixel*/
    lv_coord_t map_width = lv_area_get_width(cords_p);
    if(cords_p->y1 < masked_a.y1) {
        map_p += (uint32_t) map_width * ((masked_a.y1 - cords_p->y1)) * px_size_byte;
    }
    if(cords_p->x1 < masked_a.x1) {
        map_p += (masked_a.x1 - cords_p->x1) * px_size_byte;
    }

    /*Stores coordinates relative to the current VDB*/
    masked_a.x1 = masked_a.x1 - vdb_p->area.x1;
    masked_a.y1 = masked_a.y1 - vdb_p->area.y1;
    masked_a.x2 = masked_a.x2 - vdb_p->area.x1;
    masked_a.y2 = masked_a.y2 - vdb_p->area.y1;

    lv_coord_t vdb_width = lv_area_get_width(&vdb_p->area);
    lv_color_t * vdb_buf_tmp = vdb_p->buf;
    vdb_buf_tmp += (uint32_t) vdb_width * masked_a.y1; /*Move to the first row*/
    vdb_buf_tmp += (uint32_t) masked_a.x1; /*Move to the first col*/

    lv_coord_t row;
    lv_coord_t map_useful_w = lv_area_get_width(&masked_a);

    /*The simplest case just copy the pixels into the VDB*/
    if(chroma_key == false && alpha_byte == false && opa == LV_OPA_COVER && recolor_opa == LV_OPA_TRANSP) {

        for(row = masked_a.y1; row <= masked_a.y2; row++) {
#if USE_LV_GPU
            if(lv_disp_is_mem_blend_supported() == false) {
                sw_mem_blend(vdb_buf_tmp, (lv_color_t *)map_p, map_useful_w, opa);
            } else {
                lv_disp_mem_blend(vdb_buf_tmp, (lv_color_t *)map_p, map_useful_w, opa);
            }
#else
            sw_mem_blend(vdb_buf_tmp, (lv_color_t *)map_p, map_useful_w, opa);
#endif
            map_p += map_width * px_size_byte;               /*Next row on the map*/
            vdb_buf_tmp += vdb_width;                        /*Next row on the VDB*/
