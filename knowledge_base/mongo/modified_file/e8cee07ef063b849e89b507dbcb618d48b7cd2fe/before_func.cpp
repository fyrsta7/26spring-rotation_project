    }

    void aboutToDeleteForSharding( const Database* db, const NamespaceDetails* nsd, const DiskLoc& dl ) {
        if ( nsd->isCapped() )
            return;
        migrateFromStatus.aboutToDelete( db , dl );
    }

    class TransferModsCommand : public ChunkCommandHelper {
    public:
        TransferModsCommand() : ChunkCommandHelper( "_transferMods" ) {}

        bool run(const string& , BSONObj& cmdObj, int, string& errmsg, BSONObjBuilder& result, bool) {
            return migrateFromStatus.transferMods( errmsg, result );
        }
    } transferModsCommand;


    class InitialCloneCommand : public ChunkCommandHelper {
    public:
        InitialCloneCommand() : ChunkCommandHelper( "_migrateClone" ) {}

        bool run(const string& , BSONObj& cmdObj, int, string& errmsg, BSONObjBuilder& result, bool) {
            return migrateFromStatus.clone( errmsg, result );
        }
    } initialCloneCommand;


    /**
     * this is the main entry for moveChunk
     * called to initial a move
     * usually by a mongos
     * this is called on the "from" side
     */
    class MoveChunkCommand : public Command {
    public:
        MoveChunkCommand() : Command( "moveChunk" ) {}
        virtual void help( stringstream& help ) const {
            help << "should not be calling this directly";
        }

        virtual bool slaveOk() const { return false; }
        virtual bool adminOnly() const { return true; }
        virtual LockType locktype() const { return NONE; }


        bool run(const string& , BSONObj& cmdObj, int, string& errmsg, BSONObjBuilder& result, bool) {
            // 1. parse options
            // 2. make sure my view is complete and lock
            // 3. start migrate
            //    in a read lock, get all DiskLoc and sort so we can do as little seeking as possible
            //    tell to start transferring
            // 4. pause till migrate caught up
            // 5. LOCK
            //    a) update my config, essentially locking
            //    b) finish migrate
            //    c) update config server
            //    d) logChange to config server
            // 6. wait for all current cursors to expire
            // 7. remove data locally

            // -------------------------------

            // 1.
            string ns = cmdObj.firstElement().str();
            string to = cmdObj["to"].str();
            string from = cmdObj["from"].str(); // my public address, a tad redundant, but safe

            // fromShard and toShard needed so that 2.2 mongos can interact with either 2.0 or 2.2 mongod
            if( cmdObj["fromShard"].type() == String ){
                from = cmdObj["fromShard"].String();
            }

            if( cmdObj["toShard"].type() == String ){
                to = cmdObj["toShard"].String();
            }
            
            // if we do a w=2 after very write
            bool secondaryThrottle = cmdObj["secondaryThrottle"].trueValue();
            if ( secondaryThrottle && ! anyReplEnabled() ) {
                secondaryThrottle = false;
                warning() << "secondaryThrottle selected but no replication" << endl;
            }

            BSONObj min  = cmdObj["min"].Obj();
            BSONObj max  = cmdObj["max"].Obj();
            BSONElement shardId = cmdObj["shardId"];
            BSONElement maxSizeElem = cmdObj["maxChunkSizeBytes"];

            if ( ns.empty() ) {
                errmsg = "need to specify namespace in command";
                return false;
            }

            if ( to.empty() ) {
                errmsg = "need to specify shard to move chunk to";
                return false;
            }
            if ( from.empty() ) {
                errmsg = "need to specify shard to move chunk from";
                return false;
            }

            if ( min.isEmpty() ) {
                errmsg = "need to specify a min";
                return false;
            }

            if ( max.isEmpty() ) {
                errmsg = "need to specify a max";
                return false;
            }

            if ( shardId.eoo() ) {
                errmsg = "need shardId";
                return false;
            }

            if ( maxSizeElem.eoo() || ! maxSizeElem.isNumber() ) {
                errmsg = "need to specify maxChunkSizeBytes";
                return false;
            }
            const long long maxChunkSize = maxSizeElem.numberLong(); // in bytes

            if ( ! shardingState.enabled() ) {
                if ( cmdObj["configdb"].type() != String ) {
                    errmsg = "sharding not enabled";
                    return false;
                }
                string configdb = cmdObj["configdb"].String();
                shardingState.enable( configdb );
                configServer.init( configdb );
            }

            MoveTimingHelper timing( "from" , ns , min , max , 6 /* steps */ , errmsg );

            // Make sure we're as up-to-date as possible with shard information
            // This catches the case where we had to previously changed a shard's host by
            // removing/adding a shard with the same name
            Shard::reloadShardInfo();

            // So 2.2 mongod can interact with 2.0 mongos, mongod needs to handle either a conn
            // string or a shard in the to/from fields.  The Shard constructor handles this,
            // eventually we should break the compatibility.

            Shard fromShard( from );
            Shard toShard( to );

            log() << "received moveChunk request: " << cmdObj << migrateLog;

            timing.done(1);

            // 2.
            
            if ( migrateFromStatus.isActive() ) {
                errmsg = "migration already in progress";
                return false;
            }

            DistributedLock lockSetup( ConnectionString( shardingState.getConfigServer() , ConnectionString::SYNC ) , ns );
            dist_lock_try dlk;

            try{
                dlk = dist_lock_try( &lockSetup , (string)"migrate-" + min.toString() );
            }
            catch( LockException& e ){
                errmsg = str::stream() << "error locking distributed lock for migration " << "migrate-" << min.toString() << causedBy( e );
                return false;
            }

            if ( ! dlk.got() ) {
                errmsg = str::stream() << "the collection metadata could not be locked with lock " << "migrate-" << min.toString();
                result.append( "who" , dlk.other() );
                return false;
            }

            BSONObj chunkInfo = BSON("min" << min << "max" << max << "from" << fromShard.getName() << "to" << toShard.getName() );
            configServer.logChange( "moveChunk.start" , ns , chunkInfo );

            ShardChunkVersion maxVersion;
            ShardChunkVersion startingVersion;
            string myOldShard;
            {
                scoped_ptr<ScopedDbConnection> conn(
                        ScopedDbConnection::getInternalScopedDbConnection(
                                shardingState.getConfigServer() ) );

                BSONObj x;
                BSONObj currChunk;
                try{
                    x = conn->get()->findOne(ConfigNS::chunk,
                                             Query(BSON(ChunkFields::ns(ns)))
                                                  .sort(BSON(ChunkFields::lastmod() << -1)));

                    currChunk = conn->get()->findOne(ConfigNS::chunk,
                                                     shardId.wrap(ChunkFields::name().c_str()));
                }
                catch( DBException& e ){
                    errmsg = str::stream() << "aborted moveChunk because could not get chunk data from config server " << shardingState.getConfigServer() << causedBy( e );
                    warning() << errmsg << endl;
                    return false;
                }

                maxVersion = ShardChunkVersion::fromBSON(x, ChunkFields::lastmod());
                verify(currChunk[ChunkFields::shard()].type());
                verify(currChunk[ChunkFields::min()].type());
                verify(currChunk[ChunkFields::max()].type());
                myOldShard = currChunk[ChunkFields::shard()].String();
                conn->done();

                BSONObj currMin = currChunk[ChunkFields::min()].Obj();
                BSONObj currMax = currChunk[ChunkFields::max()].Obj();
                if ( currMin.woCompare( min ) || currMax.woCompare( max ) ) {
                    errmsg = "boundaries are outdated (likely a split occurred)";
                    result.append( "currMin" , currMin );
                    result.append( "currMax" , currMax );
                    result.append( "requestedMin" , min );
                    result.append( "requestedMax" , max );

                    warning() << "aborted moveChunk because" <<  errmsg << ": " << min << "->" << max
                                      << " is now " << currMin << "->" << currMax << migrateLog;
                    return false;
                }

                if ( myOldShard != fromShard.getName() ) {
                    errmsg = "location is outdated (likely balance or migrate occurred)";
                    result.append( "from" , fromShard.getName() );
                    result.append( "official" , myOldShard );

                    warning() << "aborted moveChunk because " << errmsg << ": chunk is at " << myOldShard
                                      << " and not at " << fromShard.getName() << migrateLog;
                    return false;
                }

                if ( maxVersion < shardingState.getVersion( ns ) ) {
                    errmsg = "official version less than mine?";
                    maxVersion.addToBSON( result, "officialVersion" );
                    shardingState.getVersion( ns ).addToBSON( result, "myVersion" );

                    warning() << "aborted moveChunk because " << errmsg << ": official " << maxVersion
                                      << " mine: " << shardingState.getVersion(ns) << migrateLog;
                    return false;
                }

                // since this could be the first call that enable sharding we also make sure to have the chunk manager up to date
                shardingState.gotShardName( myOldShard );

                // Using the maxVersion we just found will enforce a check - if we use zero version,
                // it's possible this shard will be *at* zero version from a previous migrate and
                // no refresh will be done
                // TODO: Make this less fragile
                startingVersion = maxVersion;
                shardingState.trySetVersion( ns , startingVersion /* will return updated */ );

                log() << "moveChunk request accepted at version " << startingVersion << migrateLog;
            }

            timing.done(2);

            // 3.

            ShardChunkManagerPtr chunkManager = shardingState.getShardChunkManager( ns );
            verify( chunkManager != NULL );
            BSONObj shardKeyPattern = chunkManager->getKey();
            if ( shardKeyPattern.isEmpty() ){
                errmsg = "no shard key found";
                return false;
            }

            MigrateStatusHolder statusHolder( ns , min , max , shardKeyPattern );
            if (statusHolder.isAnotherMigrationActive()) {
                errmsg = "moveChunk is already in progress from this shard";
                return false;
            }

            {
                // this gets a read lock, so we know we have a checkpoint for mods
                if ( ! migrateFromStatus.storeCurrentLocs( maxChunkSize , errmsg , result ) )
                    return false;

                scoped_ptr<ScopedDbConnection> connTo(
                        ScopedDbConnection::getScopedDbConnection( toShard.getConnString() ) );
                BSONObj res;
                bool ok;
                try{
                    ok = connTo->get()->runCommand( "admin" ,
                                                    BSON( "_recvChunkStart" << ns <<
                                                          "from" << fromShard.getConnString() <<
                                                          "min" << min <<
                                                          "max" << max <<
                                                          "shardKeyPattern" << shardKeyPattern <<
                                                          "configServer" << configServer.modelServer() <<
                                                          "secondaryThrottle" << secondaryThrottle
                                                          ) ,
                                                    res );
                }
                catch( DBException& e ){
                    errmsg = str::stream() << "moveChunk could not contact to: shard "
                                           << to << " to start transfer" << causedBy( e );
                    warning() << errmsg << endl;
                    return false;
                }

                connTo->done();

                if ( ! ok ) {
                    errmsg = "moveChunk failed to engage TO-shard in the data transfer: ";
                    verify( res["errmsg"].type() );
                    errmsg += res["errmsg"].String();
                    result.append( "cause" , res );
                    warning() << errmsg << endl;
                    return false;
                }

            }
            timing.done( 3 );

            // 4.
            for ( int i=0; i<86400; i++ ) { // don't want a single chunk move to take more than a day
                verify( !Lock::isLocked() );
                sleepsecs( 1 );
                scoped_ptr<ScopedDbConnection> conn(
                        ScopedDbConnection::getScopedDbConnection( toShard.getConnString() ) );
                BSONObj res;
                bool ok;
                try {
                    ok = conn->get()->runCommand( "admin" , BSON( "_recvChunkStatus" << 1 ) , res );
                    res = res.getOwned();
                }
                catch( DBException& e ){
                    errmsg = str::stream() << "moveChunk could not contact to: shard " << to << " to monitor transfer" << causedBy( e );
                    warning() << errmsg << endl;
                    return false;
                }

                conn->done();

                LOG(0) << "moveChunk data transfer progress: " << res << " my mem used: " << migrateFromStatus.mbUsed() << migrateLog;

                if ( ! ok || res["state"].String() == "fail" ) {
                    warning() << "moveChunk error transferring data caused migration abort: " << res << migrateLog;
                    errmsg = "data transfer error";
                    result.append( "cause" , res );
                    return false;
                }

                if ( res["state"].String() == "steady" )
                    break;

                if ( migrateFromStatus.mbUsed() > (500 * 1024 * 1024) ) {
                    // this is too much memory for us to use for this
                    // so we're going to abort the migrate
                    scoped_ptr<ScopedDbConnection> conn(
                            ScopedDbConnection::getScopedDbConnection( toShard.getConnString() ) );

                    BSONObj res;
                    conn->get()->runCommand( "admin" , BSON( "_recvChunkAbort" << 1 ) , res );
                    res = res.getOwned();
                    conn->done();
                    error() << "aborting migrate because too much memory used res: " << res << migrateLog;
                    errmsg = "aborting migrate because too much memory used";
                    result.appendBool( "split" , true );
                    return false;
                }

                killCurrentOp.checkForInterrupt();
            }
            timing.done(4);

            // 5.
            {
                // 5.a
                // we're under the collection lock here, so no other migrate can change maxVersion or ShardChunkManager state
                migrateFromStatus.setInCriticalSection( true );
                ShardChunkVersion myVersion = maxVersion;
                myVersion.incMajor();

                {
                    Lock::DBWrite lk( ns );
                    verify( myVersion > shardingState.getVersion( ns ) );

                    // bump the chunks manager's version up and "forget" about the chunk being moved
                    // this is not the commit point but in practice the state in this shard won't until the commit it done
                    shardingState.donateChunk( ns , min , max , myVersion );
                }

                log() << "moveChunk setting version to: " << myVersion << migrateLog;

                // 5.b
                // we're under the collection lock here, too, so we can undo the chunk donation because no other state change
                // could be ongoing
                {
                    BSONObj res;
                    scoped_ptr<ScopedDbConnection> connTo(
                            ScopedDbConnection::getScopedDbConnection( toShard.getConnString(),
                                                                       10.0 ) );

                    bool ok;

                    try{
                        ok = connTo->get()->runCommand( "admin" ,
                                                        BSON( "_recvChunkCommit" << 1 ) ,
                                                        res );
                    }
                    catch( DBException& e ){
                        errmsg = str::stream() << "moveChunk could not contact to: shard " << toShard.getConnString() << " to commit transfer" << causedBy( e );
                        warning() << errmsg << endl;
                        ok = false;
                    }

                    connTo->done();

                    if ( ! ok ) {
                        {
                            Lock::DBWrite lk( ns );

                            // revert the chunk manager back to the state before "forgetting" about the chunk
                            shardingState.undoDonateChunk( ns , min , max , startingVersion );
                        }

                        log() << "moveChunk migrate commit not accepted by TO-shard: " << res
                              << " resetting shard version to: " << startingVersion << migrateLog;

                        errmsg = "_recvChunkCommit failed!";
                        result.append( "cause" , res );
                        return false;
                    }

                    log() << "moveChunk migrate commit accepted by TO-shard: " << res << migrateLog;
                }

                // 5.c

                // version at which the next highest lastmod will be set
                // if the chunk being moved is the last in the shard, nextVersion is that chunk's lastmod
                // otherwise the highest version is from the chunk being bumped on the FROM-shard
                ShardChunkVersion nextVersion;

                // we want to go only once to the configDB but perhaps change two chunks, the one being migrated and another
                // local one (so to bump version for the entire shard)
                // we use the 'applyOps' mechanism to group the two updates and make them safer
                // TODO pull config update code to a module

                BSONObjBuilder cmdBuilder;

                BSONArrayBuilder updates( cmdBuilder.subarrayStart( "applyOps" ) );
                {
                    // update for the chunk being moved
                    BSONObjBuilder op;
                    op.append( "op" , "u" );
                    op.appendBool( "b" , false /* no upserting */ );
                    op.append( "ns" , ConfigNS::chunk );

                    BSONObjBuilder n( op.subobjStart( "o" ) );
                    n.append(ChunkFields::name(), Chunk::genID(ns, min));
                    myVersion.addToBSON(n, ChunkFields::lastmod());
                    n.append(ChunkFields::ns(), ns);
                    n.append(ChunkFields::min(), min);
                    n.append(ChunkFields::max(), max);
                    n.append(ChunkFields::shard(), toShard.getName());
                    n.done();

                    BSONObjBuilder q( op.subobjStart( "o2" ) );
                    q.append(ChunkFields::name(), Chunk::genID(ns, min));
                    q.done();

                    updates.append( op.obj() );
                }

                nextVersion = myVersion;

                // if we have chunks left on the FROM shard, update the version of one of them as well
                // we can figure that out by grabbing the chunkManager installed on 5.a
                // TODO expose that manager when installing it

                ShardChunkManagerPtr chunkManager = shardingState.getShardChunkManager( ns );
                if( chunkManager->getNumChunks() > 0 ) {

                    // get another chunk on that shard
                    BSONObj lookupKey;
                    BSONObj bumpMin, bumpMax;
                    do {
                        chunkManager->getNextChunk( lookupKey , &bumpMin , &bumpMax );
                        lookupKey = bumpMin;
                    }
                    while( bumpMin == min );

                    BSONObjBuilder op;
                    op.append( "op" , "u" );
                    op.appendBool( "b" , false );
                    op.append( "ns" , ConfigNS::chunk );

                    nextVersion.incMinor();  // same as used on donateChunk
                    BSONObjBuilder n( op.subobjStart( "o" ) );
                    n.append(ChunkFields::name(), Chunk::genID(ns, bumpMin));
                    nextVersion.addToBSON(n, ChunkFields::lastmod());
                    n.append(ChunkFields::ns(), ns);
                    n.append(ChunkFields::min(), bumpMin);
                    n.append(ChunkFields::max(), bumpMax);
                    n.append(ChunkFields::shard(), fromShard.getName());
                    n.done();

                    BSONObjBuilder q( op.subobjStart( "o2" ) );
                    q.append(ChunkFields::name(), Chunk::genID(ns, bumpMin));
                    q.done();

                    updates.append( op.obj() );

                    log() << "moveChunk updating self version to: " << nextVersion << " through "
                          << bumpMin << " -> " << bumpMax << " for collection '" << ns << "'" << migrateLog;

                }
                else {

                    log() << "moveChunk moved last chunk out for collection '" << ns << "'" << migrateLog;
                }

                updates.done();

                BSONArrayBuilder preCond( cmdBuilder.subarrayStart( "preCondition" ) );
                {
                    BSONObjBuilder b;
                    b.append("ns", ConfigNS::chunk);
                    b.append("q", BSON("query" << BSON(ChunkFields::ns(ns)) <<
                                       "orderby" << BSON(ChunkFields::lastmod() << -1)));
                    {
                        BSONObjBuilder bb( b.subobjStart( "res" ) );
                        // TODO: For backwards compatibility, we can't yet require an epoch here
                        bb.appendTimestamp(ChunkFields::lastmod(), maxVersion.toLong());
                        bb.done();
                    }
                    preCond.append( b.obj() );
                }

                preCond.done();

                BSONObj cmd = cmdBuilder.obj();
                LOG(7) << "moveChunk update: " << cmd << migrateLog;

                bool ok = false;
                BSONObj cmdResult;
                try {
                    scoped_ptr<ScopedDbConnection> conn(
                            ScopedDbConnection::getInternalScopedDbConnection(
                                    shardingState.getConfigServer(),
                                    10.0 ) );
                    ok = conn->get()->runCommand( "config" , cmd , cmdResult );
                    conn->done();
                }
                catch ( DBException& e ) {
                    warning() << e << migrateLog;
                    ok = false;
                    BSONObjBuilder b;
                    e.getInfo().append( b );
                    cmdResult = b.obj();
                }

                if ( ! ok ) {

                    // this could be a blip in the connectivity
                    // wait out a few seconds and check if the commit request made it
                    //
                    // if the commit made it to the config, we'll see the chunk in the new shard and there's no action
                    // if the commit did not make it, currently the only way to fix this state is to bounce the mongod so
                    // that the old state (before migrating) be brought in

                    warning() << "moveChunk commit outcome ongoing: " << cmd << " for command :" << cmdResult << migrateLog;
                    sleepsecs( 10 );

                    try {
                        scoped_ptr<ScopedDbConnection> conn(
                                ScopedDbConnection::getInternalScopedDbConnection(
                                        shardingState.getConfigServer(),
                                        10.0 ) );

                        // look for the chunk in this shard whose version got bumped
                        // we assume that if that mod made it to the config, the applyOps was successful
                        BSONObj doc = conn->get()->findOne(ConfigNS::chunk,
                                                           Query(BSON(ChunkFields::ns(ns)))
                                                               .sort(BSON(ChunkFields::lastmod() << -1)));

                        ShardChunkVersion checkVersion =
                            ShardChunkVersion::fromBSON(doc[ChunkFields::lastmod()]);

                        if ( checkVersion.isEquivalentTo( nextVersion ) ) {
                            log() << "moveChunk commit confirmed" << migrateLog;

                        }
                        else {
                            error() << "moveChunk commit failed: version is at"
                                            << checkVersion << " instead of " << nextVersion << migrateLog;
                            error() << "TERMINATING" << migrateLog;
                            dbexit( EXIT_SHARDING_ERROR );
                        }

                        conn->done();

                    }
                    catch ( ... ) {
                        error() << "moveChunk failed to get confirmation of commit" << migrateLog;
                        error() << "TERMINATING" << migrateLog;
                        dbexit( EXIT_SHARDING_ERROR );
                    }
                }

                migrateFromStatus.setInCriticalSection( false );

                // 5.d
