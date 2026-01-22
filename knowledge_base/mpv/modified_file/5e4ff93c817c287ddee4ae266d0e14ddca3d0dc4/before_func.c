static int put_image(struct vf_instance_s* vf, mp_image_t *mpi){
  mp_image_t *dmpi = NULL;

  // Close all menu who requested it
  while(vf->priv->current->cl && vf->priv->current != vf->priv->root) {
    menu_t* m = vf->priv->current;
    vf->priv->current = m->parent ? m->parent :  vf->priv->root;
    menu_close(m);
  }

  // Step 1 : save the picture
  while(go2pause == 1) {
    static char delay = 0; // Hack : wait the 2 frame to be sure to show the right picture
    delay ^= 1; // after a seek
    if(!delay) break;

    if(pause_mpi && (mpi->w != pause_mpi->w || mpi->h != pause_mpi->h ||
		     mpi->imgfmt != pause_mpi->imgfmt)) {
      free_mp_image(pause_mpi);
      pause_mpi = NULL;
    }
    if(!pause_mpi)
      pause_mpi = alloc_mpi(mpi->w,mpi->h,mpi->imgfmt);
    copy_mpi(pause_mpi,mpi);
    mp_input_queue_cmd(mp_input_parse_cmd("pause"));
    go2pause = 2;
    break;
  }

  // Grab // Ungrab the keys
  if(!mp_input_key_cb && vf->priv->current->show)
    mp_input_key_cb = key_cb;
  if(mp_input_key_cb && !vf->priv->current->show)
    mp_input_key_cb = NULL;

  if(mpi->flags&MP_IMGFLAG_DIRECT)
    dmpi = mpi->priv;
  else {
    dmpi = vf_get_image(vf->next,mpi->imgfmt,
			MP_IMGTYPE_TEMP, MP_IMGFLAG_ACCEPT_STRIDE,
			mpi->w,mpi->h);
    copy_mpi(dmpi,mpi);
  }
  menu_draw(vf->priv->current,dmpi);

  return vf_next_put_image(vf,dmpi);
}
