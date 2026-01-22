    return AF_OK;
  case AF_CONTROL_VOLUME_LEVEL | AF_CONTROL_SET:
    return af_from_dB(AF_NCH,(float*)arg,s->level,20.0,-200.0,60.0);
  case AF_CONTROL_VOLUME_LEVEL | AF_CONTROL_GET:
    return af_to_dB(AF_NCH,s->level,(float*)arg,20.0);
  case AF_CONTROL_VOLUME_PROBE | AF_CONTROL_GET:
    return af_to_dB(AF_NCH,s->pow,(float*)arg,10.0);
  case AF_CONTROL_VOLUME_PROBE_MAX | AF_CONTROL_GET:
    return af_to_dB(AF_NCH,s->max,(float*)arg,10.0);
  case AF_CONTROL_PRE_DESTROY:{
    float m = 0.0;
    int i;
    if(!s->fast){
      for(i=0;i<AF_NCH;i++)
	m=max(m,s->max[i]);
	af_to_dB(1, &m, &m, 10.0);
	mp_msg(MSGT_AFILTER, MSGL_INFO, "[volume] The maximum volume was %0.2fdB \n", m);
    }
    return AF_OK;
  }
  }
  return AF_UNKNOWN;
}

// Deallocate memory
static void uninit(struct af_instance_s* af)
{
    free(af->data);
    free(af->setup);
}

// Filter data through filter
static af_data_t* play(struct af_instance_s* af, af_data_t* data)
{
  af_data_t*    c   = data;			// Current working data
  af_volume_t*  s   = (af_volume_t*)af->setup; 	// Setup for this instance
  register int	nch = c->nch;			// Number of channels
  register int  i   = 0;

  // Basic operation volume control only (used on slow machines)
  if(af->data->format == (AF_FORMAT_S16_NE)){
    int16_t*    a   = (int16_t*)c->audio;	// Audio data
    int         len = c->len/2;			// Number of samples
    for (int ch = 0; ch < nch; ch++) {
      int vol = 256.0 * s->level[ch];
      if (s->enable[ch] && vol != 256) {
	for(i=ch;i<len;i+=nch){
	  register int x = (a[i] * vol) >> 8;
	  a[i]=clamp(x,SHRT_MIN,SHRT_MAX);
	}
      }
    }
  }
  // Machine is fast and data is floating point
  else if(af->data->format == (AF_FORMAT_FLOAT_NE)){
    float*   	a   	= (float*)c->audio;	// Audio data
