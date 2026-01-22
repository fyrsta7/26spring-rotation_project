 */

static int
se_InitStream( struct stream_extractor_private* priv, stream_t* s )
{
    if( priv->extractor.pf_read ) s->pf_read = se_StreamRead;
    else                          s->pf_block = se_StreamBlock;

    s->pf_seek = se_StreamSeek;
    s->pf_control = se_StreamControl;
    s->psz_url = StreamExtractorCreateMRL( priv->extractor.source->psz_url,
                                           priv->extractor.identifier,
                                           (char const **) priv->extractor.volumes,
                                           priv->extractor.volumes_count );
    if( unlikely( !s->psz_url ) )
        return VLC_ENOMEM;

    return VLC_SUCCESS;
}

static void
se_CleanStreamExtractor( struct stream_extractor_private* priv )
{
    free( (char*)priv->extractor.identifier );
