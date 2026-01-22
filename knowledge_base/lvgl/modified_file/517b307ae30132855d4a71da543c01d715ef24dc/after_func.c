
        /*File is always read to buf, thus data can be modified.*/
        header->flags |= LV_IMAGE_FLAGS_MODIFIABLE;
    }
    else if(src_type == LV_IMAGE_SRC_SYMBOL) {
        /*The size depend on the font but it is unknown here. It should be handled outside of the
         *function*/
        header->w = 1;
        header->h = 1;
        /*Symbols always have transparent parts. Important because of cover check in the draw
         *function. The actual value doesn't matter because lv_draw_label will draw it*/
        header->cf = LV_COLOR_FORMAT_A8;
    }
    else {
        LV_LOG_WARN("Image get info found unknown src type");
        return LV_RESULT_INVALID;
    }

    /*For backward compatibility, all images are not premultiplied for now.*/
    if(header->magic != LV_IMAGE_HEADER_MAGIC) {
        header->flags &= ~LV_IMAGE_FLAGS_PREMULTIPLIED;
    }

    return LV_RESULT_OK;
}

/**
 * Decode an image from a binary file
 * @param decoder pointer to the decoder
 * @param dsc     pointer to the decoder descriptor
 * @return LV_RESULT_OK: no error; LV_RESULT_INVALID: can't open the image
 */
lv_result_t lv_bin_decoder_open(lv_image_decoder_t * decoder, lv_image_decoder_dsc_t * dsc)
{
    LV_UNUSED(decoder);

    lv_result_t res = LV_RESULT_INVALID;
    lv_fs_res_t fs_res = LV_FS_RES_UNKNOWN;
    bool use_directly = false; /*If the image is already decoded and can be used directly*/

    /*Open the file if it's a file*/
    if(dsc->src_type == LV_IMAGE_SRC_FILE) {
        /*Support only "*.bin" files*/
        if(lv_strcmp(lv_fs_get_ext(dsc->src), "bin")) return LV_RESULT_INVALID;

        /*If the file was open successfully save the file descriptor*/
        decoder_data_t * decoder_data = get_decoder_data(dsc);
        if(decoder_data == NULL) {
            return LV_RESULT_INVALID;
        }

        dsc->user_data = decoder_data;
        lv_fs_file_t * f = lv_malloc(sizeof(*f));
        if(f == NULL) {
            free_decoder_data(dsc);
            return LV_RESULT_INVALID;
        }

        fs_res = lv_fs_open(f, dsc->src, LV_FS_MODE_RD);
        if(fs_res != LV_FS_RES_OK) {
            LV_LOG_WARN("Open file failed: %d", fs_res);
            lv_free(f);
            free_decoder_data(dsc);
            return LV_RESULT_INVALID;
        }

        decoder_data->f = f;    /*Now free_decoder_data will take care of the file*/

        lv_color_format_t cf = dsc->header.cf;

        if(dsc->header.flags & LV_IMAGE_FLAGS_COMPRESSED) {
            res = decode_compressed(decoder, dsc);
        }
        else if(LV_COLOR_FORMAT_IS_INDEXED(cf)) {
            if(dsc->args.use_indexed) {
                /*Palette for indexed image and whole image of A8 image are always loaded to RAM for simplicity*/
                res = load_indexed(decoder, dsc);
            }
            else {
                res = decode_indexed(decoder, dsc);
            }
        }
        else if(LV_COLOR_FORMAT_IS_ALPHA_ONLY(cf)) {
            res = decode_alpha_only(decoder, dsc);
        }
#if LV_BIN_DECODER_RAM_LOAD
        else if(cf == LV_COLOR_FORMAT_ARGB8888      \
                || cf == LV_COLOR_FORMAT_XRGB8888   \
                || cf == LV_COLOR_FORMAT_RGB888     \
                || cf == LV_COLOR_FORMAT_RGB565     \
                || cf == LV_COLOR_FORMAT_RGB565A8   \
                || cf == LV_COLOR_FORMAT_ARGB8565) {
            res = decode_rgb(decoder, dsc);
        }
#else
        else {
            /* decode them in get_area_cb */
            res = LV_RESULT_OK;
        }
#endif
    }

    else if(dsc->src_type == LV_IMAGE_SRC_VARIABLE) {
        /*The variables should have valid data*/
        lv_image_dsc_t * image = (lv_image_dsc_t *)dsc->src;
        if(image->data == NULL) {
            return LV_RESULT_INVALID;
        }

        lv_color_format_t cf = image->header.cf;
        if(dsc->header.flags & LV_IMAGE_FLAGS_COMPRESSED) {
            res = decode_compressed(decoder, dsc);
        }
        else if(LV_COLOR_FORMAT_IS_INDEXED(cf)) {
            /*Need decoder data to store converted image*/
            decoder_data_t * decoder_data = get_decoder_data(dsc);
            if(decoder_data == NULL) {
                return LV_RESULT_INVALID;
            }

            if(dsc->args.use_indexed) {
                /*Palette for indexed image and whole image of A8 image are always loaded to RAM for simplicity*/
                res = load_indexed(decoder, dsc);
                use_directly = true; /*If draw unit supports indexed image, it can be used directly.*/
            }
            else {
                res = decode_indexed(decoder, dsc);
            }
        }
        else if(LV_COLOR_FORMAT_IS_ALPHA_ONLY(cf)) {
            if(cf == LV_COLOR_FORMAT_A8) {
                res = LV_RESULT_OK;
                use_directly = true;
                dsc->decoded = (lv_draw_buf_t *)image;
            }
            else {
                /*Alpha only image will need decoder data to store pointer to decoded image, to free it when decoder closes*/
                decoder_data_t * decoder_data = get_decoder_data(dsc);
                if(decoder_data == NULL) {
                    return LV_RESULT_INVALID;
                }

                res = decode_alpha_only(decoder, dsc);
            }
        }
        else {
            /*In case of uncompressed formats the image stored in the ROM/RAM.
             *So simply give its pointer*/

            decoder_data_t * decoder_data = get_decoder_data(dsc);
            lv_draw_buf_t * decoded;
            if(image->header.flags & LV_IMAGE_FLAGS_ALLOCATED) {
                decoded = (lv_draw_buf_t *)image;
            }
            else {
                decoded = &decoder_data->c_array;
                if(image->header.stride == 0) {
                    /*If image doesn't have stride, treat it as lvgl v8 legacy image format*/
                    lv_image_dsc_t tmp = *image;
                    tmp.header.stride = (tmp.header.w * lv_color_format_get_bpp(cf) + 7) >> 3;
                    lv_draw_buf_from_image(decoded, &tmp);
                }
                else
                    lv_draw_buf_from_image(decoded, image);
            }

            dsc->decoded = decoded;

            if(decoded->header.stride == 0) {
                /*Use the auto calculated value from decoder_info callback*/
                decoded->header.stride = dsc->header.stride;
            }

            res = LV_RESULT_OK;
            use_directly = true; /*A variable image that can be used directly.*/
        }
    }

    if(res != LV_RESULT_OK) {
        free_decoder_data(dsc);
        return res;
    }

    if(dsc->decoded == NULL) return LV_RESULT_OK; /*Need to read via get_area_cb*/

    lv_draw_buf_t * decoded = (lv_draw_buf_t *)dsc->decoded;
    if(dsc->header.flags & LV_IMAGE_FLAGS_PREMULTIPLIED) {
        lv_draw_buf_set_flag(decoded, LV_IMAGE_FLAGS_PREMULTIPLIED);
    }

    lv_draw_buf_t * adjusted = lv_image_decoder_post_process(dsc, decoded);
    if(adjusted == NULL) {
        free_decoder_data(dsc);
        return LV_RESULT_INVALID;
