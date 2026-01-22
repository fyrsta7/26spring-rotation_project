
/*****************************************************************************
 * decoder_sys_t : decoder descriptor
 *****************************************************************************/
struct decoder_sys_t
{
    /* Common part between video and audio decoder */
    int i_cat;
    int i_codec_id;
    char *psz_namecodec;
    AVCodecContext      *p_context;
    AVCodec             *p_codec;

    /* Temporary buffer for libavcodec */
    uint8_t *p_output;

    /*
     * Output properties
     */
    audio_sample_format_t aout_format;
    audio_date_t          end_date;

    /*
     *
     */
    uint8_t *p_samples;
    int     i_samples;
};

/*****************************************************************************
 * InitAudioDec: initialize audio decoder
 *****************************************************************************
 * The ffmpeg codec will be opened, some memory allocated.
 *****************************************************************************/
int E_(InitAudioDec)( decoder_t *p_dec, AVCodecContext *p_context,
                      AVCodec *p_codec, int i_codec_id, char *psz_namecodec )
{
    decoder_sys_t *p_sys;
    vlc_value_t lockval;

    var_Get( p_dec->p_libvlc, "avcodec", &lockval );

    /* Allocate the memory needed to store the decoder's structure */
    if( ( p_dec->p_sys = p_sys =
          (decoder_sys_t *)malloc(sizeof(decoder_sys_t)) ) == NULL )
    {
        msg_Err( p_dec, "out of memory" );
        return VLC_EGENERIC;
    }

    p_sys->p_context = p_context;
    p_sys->p_codec = p_codec;
    p_sys->i_codec_id = i_codec_id;
    p_sys->psz_namecodec = psz_namecodec;

    /* ***** Fill p_context with init values ***** */
    p_sys->p_context->sample_rate = p_dec->fmt_in.audio.i_rate;
    p_sys->p_context->channels = p_dec->fmt_in.audio.i_channels;
    p_sys->p_context->block_align = p_dec->fmt_in.audio.i_blockalign;
    p_sys->p_context->bit_rate = p_dec->fmt_in.i_bitrate;
    p_sys->p_context->bits_per_sample = p_dec->fmt_in.audio.i_bitspersample;

    if( ( p_sys->p_context->extradata_size = p_dec->fmt_in.i_extra ) > 0 )
    {
        int i_offset = 0;

        if( p_dec->fmt_in.i_codec == VLC_FOURCC( 'f', 'l', 'a', 'c' ) )
            i_offset = 8;

        p_sys->p_context->extradata_size -= i_offset;
        p_sys->p_context->extradata =
            malloc( p_sys->p_context->extradata_size +
                    FF_INPUT_BUFFER_PADDING_SIZE );
        memcpy( p_sys->p_context->extradata,
                (char*)p_dec->fmt_in.p_extra + i_offset,
                p_sys->p_context->extradata_size );
        memset( (char*)p_sys->p_context->extradata +
