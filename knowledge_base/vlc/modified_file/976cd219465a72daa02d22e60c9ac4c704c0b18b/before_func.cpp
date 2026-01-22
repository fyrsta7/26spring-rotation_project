
        char *psz_url;
        if( asprintf( &psz_url, "attachment://%s", p_attachment->psz_name ) != -1 ) {
            vlc_meta_SetArtURL( p_meta, psz_url );
            free( psz_url );
        }
    }
}

/**
 * Read the meta information from mp4 specific tags
 * @param tag: the mp4 tag
 * @param p_demux_meta: the demuxer meta
 * @param p_meta: the meta
 */
static void ReadMetaFromMP4( MP4::Tag* tag, demux_meta_t *p_demux_meta, vlc_meta_t* p_meta )
{
    MP4::Item list;
#define SET( keyName, metaName )                                                             \
    if( tag->itemListMap().contains(keyName) )                                               \
    {                                                                                        \
        list = tag->itemListMap()[keyName];                                                  \
        vlc_meta_Set##metaName( p_meta, list.toStringList().front().toCString( true ) );     \
    }
#define SET_EXTRA( keyName, metaName )                                                   \
    if( tag->itemListMap().contains(keyName) )                                  \
    {                                                                                \
        list = tag->itemListMap()[keyName];                                     \
        vlc_meta_AddExtra( p_meta, metaName, list.toStringList().front().toCString( true ) ); \
    }

    SET("----:com.apple.iTunes:MusicBrainz Track Id", TrackID );
    SET_EXTRA("----:com.apple.iTunes:MusicBrainz Album Id", VLC_META_EXTRA_MB_ALBUMID );

#undef SET
#undef SET_EXTRA

    if( tag->itemListMap().contains("covr") )
    {
        MP4::CoverArtList list = tag->itemListMap()["covr"].toCoverArtList();
        const char *psz_format = list[0].format() == MP4::CoverArt::PNG ? "image/png" : "image/jpeg";

        msg_Dbg( p_demux_meta, "Found embedded art (%s) is %i bytes",
                 psz_format, list[0].data().size() );

        input_attachment_t *p_attachment =
                vlc_input_attachment_New( "cover", psz_format, "cover",
                                          list[0].data().data(), list[0].data().size() );
        if( p_attachment )
        {
            TAB_APPEND_CAST( (input_attachment_t**),
                             p_demux_meta->i_attachments, p_demux_meta->attachments,
                             p_attachment );
            vlc_meta_SetArtURL( p_meta, "attachment://cover" );
        }
    }
}

/**
 * Get the tags from the file using TagLib
 * @param p_this: the demux object
 * @return VLC_SUCCESS if the operation success
 */
static int ReadMeta( vlc_object_t* p_this)
{
    vlc_mutex_locker locker (&taglib_lock);
    demux_meta_t*   p_demux_meta = (demux_meta_t *)p_this;
    vlc_meta_t*     p_meta;
    FileRef f;

    p_demux_meta->p_meta = NULL;

    char *psz_uri = input_item_GetURI( p_demux_meta->p_item );
    if( unlikely(psz_uri == NULL) )
        return VLC_ENOMEM;

    if( !b_extensions_registered )
    {
        FileRef::addFileTypeResolver( &aacresolver );
        b_extensions_registered = true;
    }

    stream_t *p_stream = vlc_access_NewMRL( p_this, psz_uri );
    free( psz_uri );
    if( p_stream == NULL )
        return VLC_EGENERIC;

    VlcIostream s( p_stream );
    f = FileRef( &s );

    if( f.isNull() )
        return VLC_EGENERIC;
    if( !f.tag() || f.tag()->isEmpty() )
        return VLC_EGENERIC;

    p_demux_meta->p_meta = p_meta = vlc_meta_New();
    if( !p_meta )
        return VLC_ENOMEM;


    // Read the tags from the file
    Tag* p_tag = f.tag();

#define SET( tag, meta )                                                       \
    if( !p_tag->tag().isNull() && !p_tag->tag().isEmpty() )                    \
        vlc_meta_Set##meta( p_meta, p_tag->tag().toCString(true) )
#define SETINT( tag, meta )                                                    \
    if( p_tag->tag() )                                                         \
    {                                                                          \
        char psz_tmp[10];                                                      \
        snprintf( psz_tmp, 10, "%d", p_tag->tag() );                           \
        vlc_meta_Set##meta( p_meta, psz_tmp );                                 \
    }

    SET( title, Title );
    SET( artist, Artist );
    SET( album, Album );
    SET( comment, Description );
    SET( genre, Genre );
    SETINT( year, Date );
    SETINT( track, TrackNum );

#undef SETINT
#undef SET

    TAB_INIT( p_demux_meta->i_attachments, p_demux_meta->attachments );

    if( APE::File* ape = dynamic_cast<APE::File*>(f.file()) )
    {
        if( ape->APETag() )
            ReadMetaFromAPE( ape->APETag(), p_demux_meta, p_meta );
    }
    else
