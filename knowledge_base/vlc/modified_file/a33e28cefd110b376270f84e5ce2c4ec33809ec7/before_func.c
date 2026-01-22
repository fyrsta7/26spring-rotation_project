
    /* Input resources */
    p->p_input_resource = input_resource_New( VLC_OBJECT( p_playlist ) );
    if( unlikely(p->p_input_resource == NULL) )
        abort();

    /* Audio output (needed for volume and device controls). */
    audio_output_t *aout = input_resource_GetAout( p->p_input_resource );
    if( aout != NULL )
        input_resource_PutAout( p->p_input_resource, aout );

    /* Initialize the shared HTTP cookie jar */
    vlc_value_t cookies;
    cookies.p_address = vlc_http_cookies_new();
    if ( likely(cookies.p_address) )
    {
        var_Create( p_playlist, "http-cookies", VLC_VAR_ADDRESS );
        var_SetChecked( p_playlist, "http-cookies", VLC_VAR_ADDRESS, cookies );
    }

    /* Thread */
    playlist_Activate (p_playlist);

    /* Add service discovery modules */
    char *mods = var_InheritString( p_playlist, "services-discovery" );
    if( mods != NULL )
    {
        char *s = mods, *m;
        while( (m = strsep( &s, " :," )) != NULL )
            playlist_ServicesDiscoveryAdd( p_playlist, m );
        free( mods );
    }

    return p_playlist;
}

/**
 * Destroy playlist.
 * This is not thread-safe. Any reference to the playlist is assumed gone.
 * (In particular, all interface and services threads must have been joined).
 *
 * \param p_playlist the playlist object
 */
