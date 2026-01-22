    input_item_node_t *res;
    input_item_node_t *parent;
    if ( vlc_media_tree_Find( m_treeItem.source->tree, m_treeItem.media.get(),
                                      &res, &parent ) == false )
        return;
    refreshMediaList( std::move( mediaSource ), res->pp_children, res->i_children, true );
}

void NetworkMediaModel::onItemAdded( MediaSourcePtr mediaSource, input_item_node_t* parent,
                                  input_item_node_t *const children[],
                                  size_t count )
{
    if ( parent->p_item == m_treeItem.media.get() )
        refreshMediaList( std::move( mediaSource ), children, count, false );
}

void NetworkMediaModel::onItemRemoved( MediaSourcePtr,
                                    input_item_node_t *const children[],
                                    size_t count )
{
    for ( auto i = 0u; i < count; ++i )
    {
        InputItemPtr p_item { children[i]->p_item };
        QMetaObject::invokeMethod(this, [this, p_item=std::move(p_item)]() {
            QUrl itemUri = QUrl::fromEncoded(p_item->psz_uri);
            auto it = std::find_if( begin( m_items ), end( m_items ), [p_item, itemUri](const Item& i) {
                return QString::compare( qfu(p_item->psz_name), i.name, Qt::CaseInsensitive ) == 0 &&
                    itemUri.scheme() == i.mainMrl.scheme();
            });
            if ( it == end( m_items ) )
                return;

