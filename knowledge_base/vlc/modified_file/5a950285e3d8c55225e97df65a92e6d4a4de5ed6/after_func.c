    {
        i_strings = id3_field_getnstrings( &p_frame->fields[1] );
        while ( i_strings > 0 )
        {
            psz_temp = id3_ucs4_latin1duplicate( id3_field_getstrings( &p_frame->fields[1], --i_strings ) );
            if ( !strcmp(p_frame->id, ID3_FRAME_GENRE ) )
            {
                int i_genre;
                char *psz_endptr;
                i_genre = strtol( psz_temp, &psz_endptr, 10 );
                if( psz_temp != psz_endptr && i_genre >= 0 && i_genre < NUM_GENRES )
                {
                    input_AddInfo( p_category, (char *)p_frame->description, ppsz_genres[atoi(psz_temp)]);
                }
                else
                {
                    input_AddInfo( p_category, (char *)p_frame->description, psz_temp );
                }
            }
            else
            {
                input_AddInfo( p_category, (char *)p_frame->description, psz_temp );
            }
            free( psz_temp ); 
        }
        i++;
    }
    id3_tag_delete( p_id3_tag );
}

/*****************************************************************************
 * ParseID3Tags: check if ID3 tags at common locations. Parse them and skip it
 * if it's at the start of the file
 ****************************************************************************/
static int ParseID3Tags( vlc_object_t *p_this )
{
    input_thread_t *p_input;
    u8  *p_peek;
    int i_size;
    int i_size2;
    stream_position_t * p_pos;

    if ( p_this->i_object_type != VLC_OBJECT_INPUT )
    {
        return( VLC_EGENERIC );
    }
    p_input = (input_thread_t *)p_this;

    msg_Dbg( p_input, "Checking for ID3 tag" );

    if ( p_input->stream.b_seekable )
    {        
        /*look for a id3v1 tag at the end of the file*/
        p_pos = malloc( sizeof( stream_position_t ) );
        if ( p_pos == 0 )
        {
            msg_Err( p_input, "no mem" );
        }
        input_Tell( p_input, p_pos );
        if ( p_pos->i_size >128 )
        {
            input_AccessReinit( p_input );
            p_input->pf_seek( p_input, p_pos->i_size - 128 );
            
            /* get 10 byte id3 header */    
            if( input_Peek( p_input, &p_peek, 10 ) < 10 )
            {
                msg_Err( p_input, "cannot peek()" );
                return( VLC_EGENERIC );
            }
            i_size2 = id3_tag_query( p_peek, 10 );
            if ( i_size2 == 128 )
            {
                /* peek the entire tag */
                if ( input_Peek( p_input, &p_peek, i_size2 ) < i_size2 )
                {
                    msg_Err( p_input, "cannot peek()" );
                    return( VLC_EGENERIC );
                }
                ParseID3Tag( p_input, p_peek, i_size2 );
            }

            /* look for id3v2.4 tag at end of file */
            /* get 10 byte id3 footer */    
            if( input_Peek( p_input, &p_peek, 128 ) < 128 )
            {
                msg_Err( p_input, "cannot peek()" );
                return( VLC_EGENERIC );
            }
            i_size2 = id3_tag_query( p_peek + 118, 10 );
            if ( i_size2 < 0  && p_pos->i_size > -i_size2 )
            {                                          /* id3v2.4 footer found */
                input_AccessReinit( p_input );
                p_input->pf_seek( p_input, p_pos->i_size + i_size2 );
                /* peek the entire tag */
                if ( input_Peek( p_input, &p_peek, i_size2 ) < i_size2 )
                {
                    msg_Err( p_input, "cannot peek()" );
