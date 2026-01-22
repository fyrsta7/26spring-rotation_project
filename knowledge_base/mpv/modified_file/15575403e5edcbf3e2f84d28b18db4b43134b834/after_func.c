                        "%x, %x, %x, %x, %x, %x, %x, %x",
               &pal[ 0], &pal[ 1], &pal[ 2], &pal[ 3],
               &pal[ 4], &pal[ 5], &pal[ 6], &pal[ 7],
               &pal[ 8], &pal[ 9], &pal[10], &pal[11],
               &pal[12], &pal[13], &pal[14], &pal[15]) == 16) {
      for (i=0; i<16; i++)
        pal[i] = vobsub_palette_to_yuv(pal[i]);
      this->auto_palette = 0;
    }
    if (!strncasecmp(ptr, "forced subs: on", 15))
      this->forced_subs_only = 1;
    if (!strncmp(ptr, "custom colors: ON, tridx: ", 26) &&
        sscanf(ptr + 26, "%x, colors: %x, %x, %x, %x",
               &tridx, cuspal+0, cuspal+1, cuspal+2, cuspal+3) == 5) {
      for (i=0; i<4; i++) {
        cuspal[i] = vobsub_rgb_to_yuv(cuspal[i]);
        if (tridx & (1 << (12-4*i)))
          cuspal[i] |= 1 << 31;
      }
      this->custom = 1;
    }
  } while ((ptr=strchr(ptr,'\n')) && *++ptr);

  free(buffer);
}

void *spudec_new_scaled(unsigned int *palette, unsigned int frame_width, unsigned int frame_height, uint8_t *extradata, int extradata_len)
{
  spudec_handle_t *this = calloc(1, sizeof(spudec_handle_t));
  if (this){
    this->orig_frame_height = frame_height;
    this->orig_frame_width  = frame_width;
    // set up palette:
    if (palette)
      memcpy(this->global_palette, palette, sizeof(this->global_palette));
    else
      this->auto_palette = 1;
    if (extradata)
      spudec_parse_extradata(this, extradata, extradata_len);
    /* XXX Although the video frame is some size, the SPU frame is
       always maximum size i.e. 720 wide and 576 or 480 high */
    // For HD files in MKV the VobSub resolution can be higher though,
    // see largeres_vobsub.mkv
    if (this->orig_frame_width <= 720 && this->orig_frame_height <= 576) {
      this->orig_frame_width = 720;
      if (this->orig_frame_height == 480 || this->orig_frame_height == 240)
        this->orig_frame_height = 480;
      else
        this->orig_frame_height = 576;
    }
  }
  else
    mp_msg(MSGT_SPUDEC,MSGL_FATAL, "FATAL: spudec_init: calloc");
