    free( p_picture->p_sys );
    free( p_picture );
}

/**
 * Destroys a picture allocated with picture_NewFromFormat().
 */
static void picture_Destroy( picture_t *p_picture )
{
    aligned_free( p_picture->p[0].p_pixels );
    free( p_picture );
}

/*****************************************************************************
 *
 *****************************************************************************/
void picture_Reset( picture_t *p_picture )
{
    /* */
    p_picture->date = VLC_TS_INVALID;
    p_picture->b_force = false;
    p_picture->b_progressive = false;
    p_picture->i_nb_fields = 2;
    p_picture->b_top_field_first = false;
    PictureDestroyContext( p_picture );
}

/*****************************************************************************
 *
 *****************************************************************************/
static int LCM( int a, int b )
{
    return a * b / GCD( a, b );
}

int picture_Setup( picture_t *p_picture, const video_format_t *restrict fmt )
{
    /* Store default values */
    p_picture->i_planes = 0;
    for( unsigned i = 0; i < VOUT_MAX_PLANES; i++ )
    {
        plane_t *p = &p_picture->p[i];
        p->p_pixels = NULL;
        p->i_pixel_pitch = 0;
    }

    p_picture->i_nb_fields = 2;

    video_format_Setup( &p_picture->format, fmt->i_chroma, fmt->i_width, fmt->i_height,
                        fmt->i_visible_width, fmt->i_visible_height,
                        fmt->i_sar_num, fmt->i_sar_den );

    const vlc_chroma_description_t *p_dsc =
        vlc_fourcc_GetChromaDescription( p_picture->format.i_chroma );
    if( !p_dsc )
        return VLC_EGENERIC;

    /* We want V (width/height) to respect:
        (V * p_dsc->p[i].w.i_num) % p_dsc->p[i].w.i_den == 0
        (V * p_dsc->p[i].w.i_num/p_dsc->p[i].w.i_den * p_dsc->i_pixel_size) % 16 == 0
       Which is respected if you have
       V % lcm( p_dsc->p[0..planes].w.i_den * 16) == 0
    */
    unsigned i_modulo_w = 1;
    unsigned i_modulo_h = 1;
    unsigned i_ratio_h  = 1;

    for( unsigned i = 0; i < p_dsc->plane_count; i++ )
    {
        i_modulo_w = LCM( i_modulo_w, 16 * p_dsc->p[i].w.den );
        i_modulo_h = LCM( i_modulo_h, 16 * p_dsc->p[i].h.den );
        if( i_ratio_h < p_dsc->p[i].h.den )
            i_ratio_h = p_dsc->p[i].h.den;
