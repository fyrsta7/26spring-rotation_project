			{
				priv->list->current = i-1;
				mp_msg(MSGT_DEMUX, MSGL_V, "PROGRAM NUMBER %d: name=%s, freq=%u\n", i-1, channel->name, channel->freq);
			}
		else
		{
				mp_msg(MSGT_DEMUX, MSGL_ERR, "\n\nDVBIN: no such channel \"%s\"\n\n", progname);
	  return 0;
	}


	strcpy(priv->prev_tuning, "");
	if(!dvb_set_channel(priv, priv->card, priv->list->current))
	{
		mp_msg(MSGT_DEMUX, MSGL_ERR, "ERROR, COULDN'T SET CHANNEL  %i: ", priv->list->current);
		dvbin_close(stream);
		return 0;
	}

	mp_msg(MSGT_DEMUX, MSGL_V,  "SUCCESSFUL EXIT from dvb_streaming_start\n");

	return 1;
}




static int dvb_open(stream_t *stream, int mode, void *opts, int *file_format)
{
	// I don't force  the file format bacause, although it's almost always TS,
	// there are some providers that stream an IP multicast with M$ Mpeg4 inside
	struct stream_priv_s* p = (struct stream_priv_s*)opts;
	dvb_priv_t *priv;
	char *progname;
	int tuner_type = 0;


	if(mode != STREAM_READ)
		return STREAM_UNSUPORTED;

	stream->priv = (dvb_priv_t*) malloc(sizeof(dvb_priv_t));
	if(stream->priv ==  NULL)
		return STREAM_ERROR;

	priv = (dvb_priv_t *)stream->priv;
	priv->stream = stream;
	dvb_config = dvb_get_config();
	if(dvb_config == NULL)
	{
		free(priv);
		mp_msg(MSGT_DEMUX, MSGL_ERR, "DVB CONFIGURATION IS EMPTY, exit\n");
		return STREAM_ERROR;
	}
	dvb_config->priv = priv;
	priv->config = dvb_config;

	if(p->card < 1 || p->card > priv->config->count)
 	{
		free(priv);
		mp_msg(MSGT_DEMUX, MSGL_ERR, "NO CONFIGURATION FOUND FOR CARD N. %d, exit\n", p->card);
 		return STREAM_ERROR;
 	}
	priv->card = p->card - 1;
	priv->timeout = p->timeout;
	
	tuner_type = priv->config->cards[priv->card].type;

	if(tuner_type == 0)
	{
		free(priv);
		mp_msg(MSGT_DEMUX, MSGL_V, "OPEN_DVB: UNKNOWN OR UNDETECTABLE TUNER TYPE, EXIT\n");
		return STREAM_ERROR;
	}


	priv->tuner_type = tuner_type;

