                if ( name != "local" ) {
                    if ( only.empty() || only == name ) {
                        resyncDrop( name.c_str(), requester );
                    }
                }
            }
        }        
        syncedTo = OpTime();
        addDbNextPass.clear();
        save();
    }

    string ReplSource::resyncDrop( const char *db, const char *requester ) {
        log() << "resync: dropping database " << db << endl;
        Client::Context ctx(db);
        dropDatabase(db);
        return db;
    }
    
    /* grab initial copy of a database from the master */
    bool ReplSource::resync(string db) {
        string dummyNs = resyncDrop( db.c_str(), "internal" );
        Client::Context ctx( dummyNs );
        {
            log() << "resync: cloning database " << db << " to get an initial copy" << endl;
            ReplInfo r("resync: cloning a database");
            string errmsg;
            bool ok = cloneFrom(hostName.c_str(), errmsg, cc().database()->name, false, /*slaveok*/ true, /*replauth*/ true, /*snapshot*/false);
            if ( !ok ) {
                problem() << "resync of " << db << " from " << hostName << " failed " << errmsg << endl;
                throw SyncException();
            }
        }

        log() << "resync: done with initial clone for db: " << db << endl;

        return true;
    }

    void ReplSource::applyOperation(const BSONObj& op) {
        try {
            applyOperation_inlock( op );
        }
        catch ( UserException& e ) {
            log() << "sync: caught user assertion " << e << " while applying op: " << op << endl;;
        }
        catch ( DBException& e ) {
            log() << "sync: caught db exception " << e << " while applying op: " << op << endl;;            
        }

    }

    /* local.$oplog.main is of the form:
         { ts: ..., op: <optype>, ns: ..., o: <obj> , o2: <extraobj>, b: <boolflag> }
         ...
       see logOp() comments.
    */
    void ReplSource::sync_pullOpLog_applyOperation(BSONObj& op, OpTime *localLogTail) {
        if( logLevel >= 6 ) // op.tostring is expensive so doing this check explicitly
            log(6) << "processing op: " << op << endl;
        // skip no-op
        /* the no-op makes us process queued up databases.  so returning here would be problematic  */
////        if ( op.getStringField( "op" )[ 0 ] == 'n' )
////            return;
        
        char clientName[MaxDatabaseLen];
        const char *ns = op.getStringField("ns");
        nsToDatabase(ns, clientName);

        if ( *ns == '.' ) {
            problem() << "skipping bad op in oplog: " << op.toString() << endl;
            return;
        }
        else if ( *ns == 0 ) {
            problem() << "halting replication, bad op in oplog:\n  " << op.toString() << endl;
            replAllDead = "bad object in oplog";
            throw SyncException();
        }

        if ( !only.empty() && only != clientName )
            return;

        if( cmdLine.pretouch ) {
            if( cmdLine.pretouch > 1 ) {
                /* note: this is bad - should be put in ReplSource.  but this is first test... */
                static int countdown;
                if( countdown > 0 ) {
                    countdown--; // was pretouched on a prev pass
                    assert( countdown >= 0 );
                } else {
                    const int m = 4;
                    if( tp.get() == 0 ) {
                        int nthr = min(8, cmdLine.pretouch);
                        nthr = max(nthr, 1);
                        tp.reset( new ThreadPool(nthr) );
                    }
                    vector<BSONObj> v;
                    oplogReader.peek(v, cmdLine.pretouch);
                    unsigned a = 0;
                    while( 1 ) {
                        if( a >= v.size() ) break;
                        unsigned b = a + m - 1; // v[a..b]
                        if( b >= v.size() ) b = v.size() - 1;
                        tp->schedule(pretouchN, v, a, b);
                        DEV cout << "pretouch task: " << a << ".." << b << endl;
                        a += m;
                    }
                    // we do one too...
                    pretouchOperation(op);
                    tp->join();
                    countdown = v.size();
                }
            }
            else {
                pretouchOperation(op);
            }
        }

        dblock lk;

        if ( localLogTail && replPair && replPair->state == ReplPair::State_Master ) {
            updateSetsWithLocalOps( *localLogTail, true ); // allow unlocking
            updateSetsWithLocalOps( *localLogTail, false ); // don't allow unlocking or conversion to db backed storage
        }

        if ( replAllDead ) {
            // hmmm why is this check here and not at top of this function? does it get set between top and here?
            log() << "replAllDead, throwing SyncException: " << replAllDead << endl;
