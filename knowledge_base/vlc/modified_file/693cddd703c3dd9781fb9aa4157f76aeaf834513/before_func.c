    /* Ignore alignment if a position is given for video filter */
    if( !b_sub && p_sys->i_pos_x >= 0 && p_sys->i_pos_y >= 0 )
        p_sys->i_pos = -1;

    vlc_mutex_init( &p_sys->lock );
    LogoListLoad( VLC_OBJECT(p_filter), p_list, psz_filename );
    p_sys->b_spu_update = true;
    p_sys->b_mouse_grab = false;

    for( int i = 0; ppsz_filter_callbacks[i]; i++ )
        var_AddCallback( p_filter, ppsz_filter_callbacks[i],
                         LogoCallback, p_sys );

    /* Misc init */
    if( b_sub )
        p_filter->ops = &filter_sub_ops;
    else
        p_filter->ops = &filter_video_ops;

    free( psz_filename );
    return VLC_SUCCESS;
}

/**
 * Common close function
 */
static void Close( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys;

    for( int i = 0; ppsz_filter_callbacks[i]; i++ )
        var_DelCallback( p_filter, ppsz_filter_callbacks[i],
                         LogoCallback, p_sys );

    if( p_sys->p_blend )
        filter_DeleteBlend( p_sys->p_blend );

    LogoListUnload( &p_sys->list );
    free( p_sys );
}

/**
 * Sub source
 */
static subpicture_t *FilterSub( filter_t *p_filter, vlc_tick_t date )
{
    filter_sys_t *p_sys = p_filter->p_sys;
    logo_list_t *p_list = &p_sys->list;

    subpicture_t *p_spu;
    subpicture_region_t *p_region;
    video_format_t fmt;
    picture_t *p_pic;
    logo_t *p_logo;

    vlc_mutex_lock( &p_sys->lock );
    /* Basic test:  b_spu_update occurs on a dynamic change,
                    & i_next_pic is the general timer, when to
                    look at updating the logo image */

    if( ( !p_sys->b_spu_update && p_list->i_next_pic > date ) ||
        !p_list->i_repeat )
    {
        vlc_mutex_unlock( &p_sys->lock );
        return NULL;
    }

    /* adjust index to the next logo */
    p_logo = LogoListNext( p_list, date );
    p_sys->b_spu_update = false;

    p_pic = p_logo->p_pic;

    /* Allocate the subpicture internal data. */
    p_spu = filter_NewSubpicture( p_filter );
    if( !p_spu )
        goto exit;

    p_spu->i_start = date;
    p_spu->i_stop = 0;
    p_spu->b_ephemer = true;

    /* Send an empty subpicture to clear the display when needed */
    if( p_list->i_repeat != -1 && p_list->i_counter == 0 )
    {
        p_list->i_repeat--;
        if( p_list->i_repeat < 0 )
            goto exit;
    }
    if( !p_pic || !p_logo->i_alpha ||
        ( p_logo->i_alpha == -1 && !p_list->i_alpha ) )
        goto exit;

    /* Create new SPU region */
