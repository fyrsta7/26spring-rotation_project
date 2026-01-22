    /* Create and arm the timer */
    if( vlc_timer_create( &p_sys->timer, Fetch, p_filter ) )
        goto error;
    vlc_timer_schedule_asap( p_sys->timer, vlc_tick_from_sec(i_ttl) );

    free( psz_urls );
    return VLC_SUCCESS;

error:
    if( p_sys->p_style )
        text_style_Delete( p_sys->p_style );
    free( p_sys->psz_marquee );
    free( psz_urls );
    free( p_sys );
    return VLC_ENOMEM;
}
/*****************************************************************************
 * DestroyFilter: destroy RSS video filter
 *****************************************************************************/
static void DestroyFilter( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys;

    vlc_timer_destroy( p_sys->timer );

    text_style_Delete( p_sys->p_style );
    free( p_sys->psz_marquee );
    FreeRSS( p_sys->p_feeds, p_sys->i_feeds );
    free( p_sys );
}

static void switchToNextFeed(filter_sys_t *p_sys)
{
    p_sys->i_cur_feed = (p_sys->i_cur_feed + 1)%p_sys->i_feeds;
}

/****************************************************************************
 * Filter: the whole thing
 ****************************************************************************
 * This function outputs subpictures at regular time intervals.
 ****************************************************************************/
static subpicture_t *Filter( filter_t *p_filter, vlc_tick_t date )
{
    filter_sys_t *p_sys = p_filter->p_sys;
    subpicture_t *p_spu;
    video_format_t fmt;
    subpicture_region_t *p_region;

    int i_item;
    rss_feed_t *p_feed;

    vlc_mutex_lock( &p_sys->lock );

    /* Check if the feeds have been fetched and that we have some feeds */
    if( !p_sys->b_fetched && p_sys->i_feeds > 0 )
    {
        vlc_mutex_unlock( &p_sys->lock );
        return NULL;
    }

    /* If the current feed has no item then switch to the next feed
       and skip further processing */
    if (p_sys->p_feeds[p_sys->i_cur_feed].i_items == 0)
    {
        switchToNextFeed(p_sys);
        vlc_mutex_unlock( &p_sys->lock );
        return NULL;
    }

    if( p_sys->last_date
       + ( p_sys->i_cur_char <= 0 &&
           p_sys->i_cur_item == ( p_sys->i_title == scroll_title ? -1 : 0 ) ? 5 : 1 )
           /* ( ... ? 5 : 1 ) means "wait 5 times more for the 1st char" */
       * p_sys->i_speed > date )
    {
        vlc_mutex_unlock( &p_sys->lock );
        return NULL;
    }

    p_sys->last_date = date;
    p_sys->i_cur_char++;

    if( p_sys->i_cur_item == -1 ?
            p_sys->p_feeds[p_sys->i_cur_feed].psz_title[p_sys->i_cur_char] == 0 :
            p_sys->p_feeds[p_sys->i_cur_feed].p_items[p_sys->i_cur_item].psz_title[p_sys->i_cur_char] == 0 )
    {
        p_sys->i_cur_char = 0;
        p_sys->i_cur_item++;
        if( p_sys->i_cur_item >= p_sys->p_feeds[p_sys->i_cur_feed].i_items )
        {
            if( p_sys->i_title == scroll_title )
                p_sys->i_cur_item = -1;
            else
                p_sys->i_cur_item = 0;
            switchToNextFeed(p_sys);
        }
    }

    p_spu = filter_NewSubpicture( p_filter );
    if( !p_spu )
    {
        vlc_mutex_unlock( &p_sys->lock );
        return NULL;
    }

    video_format_Init( &fmt, VLC_CODEC_TEXT );

    p_spu->p_region = subpicture_region_New( &fmt );
    if( !p_spu->p_region )
    {
        subpicture_Delete( p_spu );
        vlc_mutex_unlock( &p_sys->lock );
        return NULL;
    }

    /* Generate the string that will be displayed. This string is supposed to
       be p_sys->i_length characters long. */
    i_item = p_sys->i_cur_item;
    p_feed = &p_sys->p_feeds[p_sys->i_cur_feed];
    char *feed_title = p_feed->psz_title;
    char *item_title = p_feed->p_items[i_item].psz_title;

    if( ( p_feed->p_pic && p_sys->i_title == default_title )
        || p_sys->i_title == hide_title )
    {
        /* Don't display the feed's title if we have an image */
        snprintf( p_sys->psz_marquee, p_sys->i_length, "%s",
                  item_title + p_sys->i_cur_char );
    }
    else if( ( !p_feed->p_pic && p_sys->i_title == default_title )
             || p_sys->i_title == prepend_title )
    {
        snprintf( p_sys->psz_marquee, p_sys->i_length, "%s : %s",
                  feed_title,
                  item_title + p_sys->i_cur_char );
    }
    else /* scrolling title */
    {
        if( i_item == -1 )
        {
            snprintf( p_sys->psz_marquee, p_sys->i_length, "%s : %s",
                      feed_title + p_sys->i_cur_char,
                      p_feed->p_items[i_item+1].psz_title );
            // Set i_item to 0 as the first item title was already printed.
            i_item = 0;
        }
        else
            snprintf( p_sys->psz_marquee, p_sys->i_length, "%s",
                      item_title + p_sys->i_cur_char );
    }

    while( strlen( p_sys->psz_marquee ) < (unsigned int)p_sys->i_length )
    {
        i_item++;
        if( i_item == p_feed->i_items ) break;
        snprintf( strchr( p_sys->psz_marquee, 0 ),
                  p_sys->i_length - strlen( p_sys->psz_marquee ),
                  " - %s",
                  p_feed->p_items[i_item].psz_title );
    }

    /* Calls to snprintf might split multibyte UTF8 chars ...
     * which freetype doesn't like. */
    {
        char *a = strdup( p_sys->psz_marquee );
        char *a2 = a;
        char *b = p_sys->psz_marquee;
        EnsureUTF8( p_sys->psz_marquee );
        /* we want to use ' ' instead of '?' for erroneous chars */
        while( *b != '\0' )
        {
            if( *b != *a ) *b = ' ';
            b++;a++;
        }
        free( a2 );
    }

    p_spu->p_region->p_text = text_segment_New(p_sys->psz_marquee);
    if( p_sys->p_style->i_font_size > 0 )
        p_spu->p_region->fmt.i_visible_height = p_sys->p_style->i_font_size;
    p_spu->i_start = date;
    p_spu->i_stop  = 0;
    p_spu->b_ephemer = true;

    /*  where to locate the string: */
    if( p_sys->i_pos < 0 )
    {   /*  set to an absolute xy */
        p_spu->p_region->i_align = SUBPICTURE_ALIGN_LEFT | SUBPICTURE_ALIGN_TOP;
        p_spu->b_absolute = true;
    }
    else
