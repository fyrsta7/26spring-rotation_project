        void _update( Request& r , DbMessage& d, ChunkManagerPtr manager ){
            int flags = d.pullInt();
            
            BSONObj query = d.nextJsObj();
            uassert( 10201 ,  "invalid update" , d.moreJSObjs() );
            BSONObj toupdate = d.nextJsObj();

            BSONObj chunkFinder = query;
            
            bool upsert = flags & UpdateOption_Upsert;
            bool multi = flags & UpdateOption_Multi;

            if ( multi )
                uassert( 10202 ,  "can't mix multi and upsert and sharding" , ! upsert );

            if ( upsert && !(manager->hasShardKey(toupdate) ||
                             (toupdate.firstElement().fieldName()[0] == '$' && manager->hasShardKey(query))))
            {
                throw UserException( 8012 , "can't upsert something without shard key" );
            }

            bool save = false;
            if ( ! manager->hasShardKey( query ) ){
                if ( multi ){
                }
                else if ( strcmp( query.firstElement().fieldName() , "_id" ) || query.nFields() != 1 ){
                    throw UserException( 8013 , "can't do update with query that doesn't have the shard key" );
                }
                else {
                    save = true;
                    chunkFinder = toupdate;
                }
            }

            
            if ( ! save ){
                if ( toupdate.firstElement().fieldName()[0] == '$' ){
                    BSONObjIterator ops(toupdate);
                    while(ops.more()){
                        BSONElement op(ops.next());
                        if (op.type() != Object)
                            continue;
                        BSONObjIterator fields(op.embeddedObject());
                        while(fields.more()){
                            const string field = fields.next().fieldName();
                            uassert(13123, "Can't modify shard key's value", ! manager->getShardKey().partOfShardKey(field));
                        }
                    }
                } else if ( manager->hasShardKey( toupdate ) ){
                    uassert( 8014, "change would move shards!", manager->getShardKey().compare( query , toupdate ) == 0 );
                } else {
                    uasserted(12376, "shard key must be in update object");
                }
            }
            
            if ( multi ){
                set<Shard> shards;
                manager->getShardsForQuery( shards , chunkFinder );
                for ( set<Shard>::iterator i=shards.begin(); i!=shards.end(); i++){
                    doWrite( dbUpdate , r , *i );
                }
            }
            else {
                ChunkPtr c = manager->findChunk( chunkFinder );
                doWrite( dbUpdate , r , c->getShard() );
                c->splitIfShould( d.msg().header()->dataLen() );
            }

        }