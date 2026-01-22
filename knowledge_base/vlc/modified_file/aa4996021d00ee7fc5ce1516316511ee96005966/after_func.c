    if( stream == NULL )
        return NULL;

    char const* anchor = strchr( mrl, '#' );

    if( anchor == NULL )
        return stream;

    char const* extra;
    if( stream_extractor_AttachParsed( &stream, anchor + 1, &extra ) )
    {
        msg_Err( parent, "unable to open %s", mrl );
        vlc_stream_Delete( stream );
        return NULL;
    }

    if( extra && *extra )
        msg_Warn( parent, "ignoring extra fragment data: %s", extra );

    return stream;
}

/**
 * Read from the stream until first newline.
 * \param s Stream handle to read from
 * \return A pointer to the allocated output string. You need to free this when you are done.
 */
#define STREAM_PROBE_LINE 2048
#define STREAM_LINE_MAX (2048*100)
char *vlc_stream_ReadLine( stream_t *s )
{
    stream_priv_t *priv = (stream_priv_t *)s;

    /* Let's fail quickly if this is a readdir access */
    if( s->pf_read == NULL && s->pf_block == NULL )
        return NULL;

    /* BOM detection */
    if( vlc_stream_Tell( s ) == 0 )
    {
        const uint8_t *p_data;
        ssize_t i_data = vlc_stream_Peek( s, &p_data, 2 );

        if( i_data <= 0 )
            return NULL;

        if( unlikely(priv->text.conv != (vlc_iconv_t)-1) )
        {   /* seek back to beginning? reset */
            vlc_iconv_close( priv->text.conv );
            priv->text.conv = (vlc_iconv_t)-1;
        }
        priv->text.char_width = 1;
        priv->text.little_endian = false;

        if( i_data >= 2 )
        {
            const char *psz_encoding = NULL;
            bool little_endian = false;

            if( !memcmp( p_data, "\xFF\xFE", 2 ) )
            {
                psz_encoding = "UTF-16LE";
                little_endian = true;
            }
            else if( !memcmp( p_data, "\xFE\xFF", 2 ) )
            {
                psz_encoding = "UTF-16BE";
            }

            /* Open the converter if we need it */
            if( psz_encoding != NULL )
            {
                msg_Dbg( s, "UTF-16 BOM detected" );
                priv->text.conv = vlc_iconv_open( "UTF-8", psz_encoding );
                if( unlikely(priv->text.conv == (vlc_iconv_t)-1) )
                {
                    msg_Err( s, "iconv_open failed" );
                    return NULL;
                }
                priv->text.char_width = 2;
                priv->text.little_endian = little_endian;
            }
        }
    }

    size_t i_line = 0;
    const uint8_t *p_data;

    for( ;; )
    {
        size_t i_peek = i_line == 0 ? STREAM_PROBE_LINE
                                    : __MIN( i_line * 2, STREAM_LINE_MAX );

        /* Probe more data */
        ssize_t i_data = vlc_stream_Peek( s, &p_data, i_peek );
        if( i_data <= 0 )
            return NULL;

        /* Deal here with lone-byte incomplete UTF-16 sequences at EOF
           that we won't be able to process anyway */
        if( i_data < priv->text.char_width )
        {
            assert( priv->text.char_width == 2 );
            uint8_t inc;
            ssize_t i_inc = vlc_stream_Read( s, &inc, priv->text.char_width );
            assert( i_inc == i_data );
            if( i_inc > 0 )
                msg_Err( s, "discarding incomplete UTF-16 sequence at EOF: 0x%02x", inc );
            return NULL;
        }

        /* Keep to text encoding character width boundary */
        if( i_data % priv->text.char_width )
            i_data = i_data - ( i_data % priv->text.char_width );

        if( (size_t) i_data == i_line )
            break; /* No more data */

        assert( (size_t) i_data > i_line );

        /* Resume search for an EOL where we left off */
        const uint8_t *p_cur = p_data + i_line, *psz_eol;

        /* FIXME: <CR> behavior varies depending on where buffer
           boundaries happen to fall; a <CR><LF> across the boundary
           creates a bogus empty line. */
        if( priv->text.char_width == 1 )
        {
            /* UTF-8: 0A <LF> */
            psz_eol = memchr( p_cur, '\n', i_data - i_line );
            if( psz_eol == NULL )
                /* UTF-8: 0D <CR> */
                psz_eol = memchr( p_cur, '\r', i_data - i_line );
        }
        else
        {
            const uint8_t *p_last = p_data + i_data - priv->text.char_width;
            uint16_t eol = priv->text.little_endian ? 0x0A00 : 0x000A;

            assert( priv->text.char_width == 2 );
            psz_eol = NULL;
            /* UTF-16: 000A <LF> */
            for( const uint8_t *p = p_cur; p <= p_last; p += 2 )
            {
                if( U16_AT( p ) == eol )
                {
                     psz_eol = p;
                     break;
                }
            }

            if( psz_eol == NULL )
            {   /* UTF-16: 000D <CR> */
                eol = priv->text.little_endian ? 0x0D00 : 0x000D;
                for( const uint8_t *p = p_cur; p <= p_last; p += 2 )
                {
                    if( U16_AT( p ) == eol )
                    {
                        psz_eol = p;
                        break;
                    }
                }
            }
        }

        if( psz_eol )
        {
            i_line = (psz_eol - p_data) + priv->text.char_width;
            /* We have our line */
            break;
        }

        i_line = i_data;

        if( i_line >= STREAM_LINE_MAX )
        {
            msg_Err( s, "line too long, exceeding %zu bytes",
                     (size_t) STREAM_LINE_MAX );
            return NULL;
        }
    }

    if( i_line == 0 ) /* We failed to read any data, probably EOF */
        return NULL;

    /* If encoding conversion is required, UTF-8 needs at most 150%
       as long a buffer as UTF-16 */
    size_t i_line_conv = priv->text.char_width == 1 ? i_line : i_line * 3 / 2;
    char *p_line = malloc( i_line_conv + 1 ); /* +1 for easy \0 append */
    if( !p_line )
        return NULL;
    void *p_read = p_line;

    if( priv->text.char_width > 1 )
    {
        size_t i_in = i_line, i_out = i_line_conv;
        const char * p_in = (char *) p_data;
        char * p_out = p_line;

        if( vlc_iconv( priv->text.conv, &p_in, &i_in, &p_out, &i_out ) == VLC_ICONV_ERR )
        {
