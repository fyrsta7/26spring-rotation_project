    }

    ChunkPtr ChunkManager::findChunkOnServer( const Shard& shard ) const {
        rwlock lk( _lock , false ); 
 
        for ( ChunkMap::const_iterator i=_chunkMap.begin(); i!=_chunkMap.end(); ++i ){
            ChunkPtr c = i->second;
            if ( c->getShard() == shard )
                return c;
        }

        return ChunkPtr();
    }

    void ChunkManager::getShardsForQuery( set<Shard>& shards , const BSONObj& query ){
        rwlock lk( _lock , false ); 

        //TODO look into FieldRangeSetOr
        FieldRangeSet frs(_ns.c_str(), query, false);
        uassert(13088, "no support for special queries yet", frs.getSpecial().empty());

        {
            // special case if most-significant field isn't in query
            FieldRange range = frs.range(_key.key().firstElement().fieldName());
            if (!range.nontrivial()){
                getAllShards(shards);
                return;
            }
        }

        BoundList ranges = frs.indexBounds(_key.key(), 1);
        for (BoundList::const_iterator it=ranges.begin(), end=ranges.end(); it != end; ++it){
            BSONObj minObj = it->first.replaceFieldNames(_key.key());
            BSONObj maxObj = it->second.replaceFieldNames(_key.key());

            ChunkRangeMap::const_iterator min, max;
