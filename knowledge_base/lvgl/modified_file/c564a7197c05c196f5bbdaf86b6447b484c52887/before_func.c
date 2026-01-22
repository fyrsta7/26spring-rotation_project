    }

    if(rlottie->task) {
        lv_timer_del(rlottie->task);
        rlottie->task = NULL;
        rlottie->play_ctrl = LV_RLOTTIE_CTRL_FORWARD;
        rlottie->dest_frame = 0;
    }

    lv_img_cache_invalidate_src(&rlottie->imgdsc);
    if(rlottie->allocated_buf) {
        lv_free(rlottie->allocated_buf);
        rlottie->allocated_buf = NULL;
        rlottie->allocated_buffer_size = 0;
    }

}

