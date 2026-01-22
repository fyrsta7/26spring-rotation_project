    var_AddCallback( p_playlist, "playlist-item-append", AllCallback, p_intf );
    var_AddCallback( p_playlist, "playlist-item-deleted", AllCallback, p_intf );
    var_AddCallback( p_playlist, "random", AllCallback, p_intf );
    var_AddCallback( p_playlist, "repeat", AllCallback, p_intf );
    var_AddCallback( p_playlist, "loop", AllCallback, p_intf );
    PL_UNLOCK;
    pl_Release( p_intf );

    UpdateCaps( p_intf );

    return VLC_SUCCESS;
}

/*****************************************************************************
 * Close: destroy interface
 *****************************************************************************/

static void Close   ( vlc_object_t *p_this )
{
    intf_thread_t   *p_intf     = (intf_thread_t*) p_this;
    playlist_t      *p_playlist = pl_Hold( p_intf );;
    input_thread_t  *p_input;

    var_DelCallback( p_playlist, "item-current", AllCallback, p_intf );
    var_DelCallback( p_playlist, "intf-change", AllCallback, p_intf );
    var_DelCallback( p_playlist, "playlist-item-append", AllCallback, p_intf );
    var_DelCallback( p_playlist, "playlist-item-deleted", AllCallback, p_intf );
    var_DelCallback( p_playlist, "random", AllCallback, p_intf );
    var_DelCallback( p_playlist, "repeat", AllCallback, p_intf );
    var_DelCallback( p_playlist, "loop", AllCallback, p_intf );

    p_input = playlist_CurrentInput( p_playlist );
    if ( p_input )
    {
        var_DelCallback( p_input, "state", AllCallback, p_intf );
        vlc_object_release( p_input );
    }

    pl_Release( p_intf );

    dbus_connection_unref( p_intf->p_sys->p_conn );

    // Free the events array
    for( int i = 0; i < vlc_array_count( p_intf->p_sys->p_events ); i++ )
    {
        callback_info_t* info = vlc_array_item_at_index( p_intf->p_sys->p_events, i );
        free( info );
