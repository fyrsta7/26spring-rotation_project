
    DEL_CB( borderw );
    DEL_CB( borderh );
    DEL_CB( rows );
    DEL_CB( cols );
    DEL_CB( alpha );
    DEL_CB( position );
    DEL_CB( delay );

    DEL_CB( keep-aspect-ratio );
    DEL_CB( order );
#undef DEL_CB

    if( !p_sys->b_keep )
    {
        image_HandlerDelete( p_sys->p_image );
    }

    if( p_sys->i_order_length )
    {
        for( int i_index = 0; i_index < p_sys->i_order_length; i_index++ )
        {
            free( p_sys->ppsz_order[i_index] );
        }
        free( p_sys->ppsz_order );
    }
    if( p_sys->i_offsets_length )
    {
        free( p_sys->pi_x_offsets );
        free( p_sys->pi_y_offsets );
        p_sys->i_offsets_length = 0;
    }

    free( p_sys );
}

/*****************************************************************************
 * Filter
 *****************************************************************************/
static subpicture_t *Filter( filter_t *p_filter, vlc_tick_t date )
{
    filter_sys_t *p_sys = p_filter->p_sys;
    bridge_t *p_bridge;

    int i_real_index, i_row, i_col;
    int i_greatest_real_index_used = p_sys->i_order_length - 1;

    unsigned int col_inner_width, row_inner_height;

    subpicture_region_t *p_region;
    subpicture_region_t *p_region_prev = NULL;

    /* Allocate the subpicture internal data. */
    subpicture_t *p_spu = filter_NewSubpicture( p_filter );
    if( !p_spu )
        return NULL;

    /* Initialize subpicture */
    p_spu->i_channel = 0;
    p_spu->i_start  = date;
    p_spu->i_stop = 0;
    p_spu->b_ephemer = true;
    p_spu->i_alpha = p_sys->i_alpha;
    p_spu->b_absolute = false;

    p_spu->i_original_picture_width = p_sys->i_width;
    p_spu->i_original_picture_height = p_sys->i_height;

    vlc_mutex_lock( &p_sys->lock );
    vlc_global_lock( VLC_MOSAIC_MUTEX );

    p_bridge = GetBridge( p_filter );
    if ( p_bridge == NULL )
    {
        vlc_global_unlock( VLC_MOSAIC_MUTEX );
        vlc_mutex_unlock( &p_sys->lock );
        return p_spu;
    }

    if ( p_sys->i_position == position_offsets )
    {
        /* If we have either too much or not enough offsets, fall-back
         * to automatic positioning. */
        if ( p_sys->i_offsets_length != p_sys->i_order_length )
        {
            msg_Err( p_filter,
                     "Number of specified offsets (%d) does not match number "
                     "of input substreams in mosaic-order (%d), falling back "
                     "to mosaic-position=0",
                     p_sys->i_offsets_length, p_sys->i_order_length );
            p_sys->i_position = position_auto;
        }
    }

    if ( p_sys->i_position == position_auto )
    {
        int i_numpics = p_sys->i_order_length; /* keep slots and all */
        for( int i_index = 0; i_index < p_bridge->i_es_num; i_index++ )
        {
            bridged_es_t *p_es = p_bridge->pp_es[i_index];
            if ( !p_es->b_empty )
            {
                i_numpics ++;
                if( p_sys->i_order_length && p_es->psz_id != NULL )
                {
                    /* We also want to leave slots for images given in
                     * mosaic-order that are not available in p_vout_picture */
                    for( int i = 0; i < p_sys->i_order_length ; i++ )
                    {
                        if( !strcmp( p_sys->ppsz_order[i], p_es->psz_id ) )
                        {
                            i_numpics--;
                            break;
                        }
                    }

                }
            }
        }
        p_sys->i_rows = ceil(sqrt( (double)i_numpics ));
        p_sys->i_cols = ( i_numpics % p_sys->i_rows == 0 ?
                            i_numpics / p_sys->i_rows :
                            i_numpics / p_sys->i_rows + 1 );
    }

    col_inner_width  = ( ( p_sys->i_width - ( p_sys->i_cols - 1 )
                       * p_sys->i_borderw ) / p_sys->i_cols );
    row_inner_height = ( ( p_sys->i_height - ( p_sys->i_rows - 1 )
                       * p_sys->i_borderh ) / p_sys->i_rows );

    i_real_index = 0;

    for( int i_index = 0; i_index < p_bridge->i_es_num; i_index++ )
    {
        bridged_es_t *p_es = p_bridge->pp_es[i_index];
        video_format_t fmt_in, fmt_out;
        picture_t *p_converted;

        if ( p_es->b_empty )
            continue;

        while ( !vlc_picture_chain_IsEmpty( &p_es->pictures ) )
        {
            picture_t *front = vlc_picture_chain_PeekFront( &p_es->pictures );
            if ( front->date + p_sys->i_delay >= date )
                break; // front picture not late

            if ( vlc_picture_chain_HasNext( &p_es->pictures ) )
            {
                // front picture is late and has more pictures chained, skip it
                front = vlc_picture_chain_PopFront( &p_es->pictures );
                picture_Release( front );
                continue;
            }

            if ( front->date + p_sys->i_delay + BLANK_DELAY < date )
            {
                // front picture is late and too old, don't display it
                front = vlc_picture_chain_PopFront( &p_es->pictures );
                // the picture chain is empty as the front didn't have chained pics
                picture_Release( front );
                break;
            }
            else
            {
                // front picture is a little late, display it
                msg_Dbg( p_filter, "too late picture for %s (%"PRId64 ")",
                         p_es->psz_id,
                         date - front->date - p_sys->i_delay );
                break;
            }
        }

        if ( vlc_picture_chain_IsEmpty( &p_es->pictures ) )
            continue;

        if ( p_sys->i_order_length == 0 )
        {
            i_real_index++;
        }
        else
        {
            int i;
            for ( i = 0; i <= p_sys->i_order_length; i++ )
            {
                if ( i == p_sys->i_order_length ) break;
                if ( strcmp( p_es->psz_id, p_sys->ppsz_order[i] ) == 0 )
                {
                    i_real_index = i;
                    break;
                }
            }
            if ( i == p_sys->i_order_length )
                i_real_index = ++i_greatest_real_index_used;
        }
        i_row = ( i_real_index / p_sys->i_cols ) % p_sys->i_rows;
        i_col = i_real_index % p_sys->i_cols ;

        video_format_Init( &fmt_in, 0 );
        video_format_Init( &fmt_out, 0 );

        p_converted = vlc_picture_chain_PeekFront( &p_es->pictures );
        if ( !p_sys->b_keep )
        {
            /* Convert the images */
            fmt_in.i_chroma = p_converted->format.i_chroma;
            fmt_in.i_height = p_converted->format.i_height;
            fmt_in.i_width = p_converted->format.i_width;

            if( fmt_in.i_chroma == VLC_CODEC_YUVA ||
                fmt_in.i_chroma == VLC_CODEC_RGBA )
                fmt_out.i_chroma = VLC_CODEC_YUVA;
            else
                fmt_out.i_chroma = VLC_CODEC_I420;
            fmt_out.i_width = col_inner_width;
            fmt_out.i_height = row_inner_height;

            if( p_sys->b_ar ) /* keep aspect ratio */
            {
                if( (float)fmt_out.i_width / (float)fmt_out.i_height
                      > (float)fmt_in.i_width / (float)fmt_in.i_height )
                {
                    fmt_out.i_width = ( fmt_out.i_height * fmt_in.i_width )
                                         / fmt_in.i_height;
                }
                else
                {
                    fmt_out.i_height = ( fmt_out.i_width * fmt_in.i_height )
                                        / fmt_in.i_width;
                }
             }

            fmt_out.i_visible_width = fmt_out.i_width;
            fmt_out.i_visible_height = fmt_out.i_height;

            p_converted = image_Convert( p_sys->p_image, p_converted,
                                         &fmt_in, &fmt_out );
            if( !p_converted )
            {
                msg_Warn( p_filter,
                           "image resizing and chroma conversion failed" );
                video_format_Clean( &fmt_in );
                video_format_Clean( &fmt_out );
                continue;
            }
        }
        else
        {
            fmt_in.i_width = fmt_out.i_width = p_converted->format.i_width;
            fmt_in.i_height = fmt_out.i_height = p_converted->format.i_height;
            fmt_in.i_chroma = fmt_out.i_chroma = p_converted->format.i_chroma;
            fmt_out.i_visible_width = fmt_out.i_width;
            fmt_out.i_visible_height = fmt_out.i_height;
        }

        p_region = subpicture_region_New( &fmt_out );
        /* FIXME the copy is probably not needed anymore */
        if( p_region )
            picture_Copy( p_region->p_picture, p_converted );
        if( !p_sys->b_keep )
            picture_Release( p_converted );

        if( !p_region )
        {
            video_format_Clean( &fmt_in );
            video_format_Clean( &fmt_out );
            msg_Err( p_filter, "cannot allocate SPU region" );
            subpicture_Delete( p_spu );
            vlc_global_unlock( VLC_MOSAIC_MUTEX );
            vlc_mutex_unlock( &p_sys->lock );
            return NULL;
        }

        if( p_es->i_x >= 0 && p_es->i_y >= 0 )
        {
            p_region->i_x = p_es->i_x;
            p_region->i_y = p_es->i_y;
        }
        else if( p_sys->i_position == position_offsets )
        {
            p_region->i_x = p_sys->pi_x_offsets[i_real_index];
            p_region->i_y = p_sys->pi_y_offsets[i_real_index];
        }
        else
        {
            if( fmt_out.i_width > col_inner_width ||
                p_sys->b_ar || p_sys->b_keep )
            {
                /* we don't have to center the video since it takes the
                whole rectangle area or it's larger than the rectangle */
                p_region->i_x = p_sys->i_xoffset
                            + i_col * ( p_sys->i_width / p_sys->i_cols )
                            + ( i_col * p_sys->i_borderw ) / p_sys->i_cols;
            }
            else
            {
                /* center the video in the dedicated rectangle */
                p_region->i_x = p_sys->i_xoffset
                        + i_col * ( p_sys->i_width / p_sys->i_cols )
                        + ( i_col * p_sys->i_borderw ) / p_sys->i_cols
                        + ( col_inner_width - fmt_out.i_width ) / 2;
