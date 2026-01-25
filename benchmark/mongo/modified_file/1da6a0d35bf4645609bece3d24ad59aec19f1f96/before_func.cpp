    void ReplSource::sync_pullOpLog_applyOperation(BSONObj& op, OpTime *localLogTail) {
        log( 6 ) << "processing op: " << op << endl;
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
            throw SyncException();
        }
        
        Client::Context ctx( ns );
        ctx.getClient()->curop()->reset();

        bool empty = ctx.db()->isEmpty();
        bool incompleteClone = incompleteCloneDbs.count( clientName ) != 0;

        if( logLevel >= 6 )
            log(6) << "ns: " << ns << ", justCreated: " << ctx.justCreated() << ", empty: " << empty << ", incompleteClone: " << incompleteClone << endl;
        
        // always apply admin command command
        // this is a bit hacky -- the semantics of replication/commands aren't well specified
        if ( strcmp( clientName, "admin" ) == 0 && *op.getStringField( "op" ) == 'c' ) {
            applyOperation( op );
            return;
        }
        
        if ( ctx.justCreated() || empty || incompleteClone ) {
            // we must add to incomplete list now that setClient has been called
            incompleteCloneDbs.insert( clientName );
            if ( nClonedThisPass ) {
                /* we only clone one database per pass, even if a lot need done.  This helps us
                 avoid overflowing the master's transaction log by doing too much work before going
                 back to read more transactions. (Imagine a scenario of slave startup where we try to
                 clone 100 databases in one pass.)
                 */
                addDbNextPass.insert( clientName );
            } else {
                if ( incompleteClone ) {
                    log() << "An earlier initial clone of '" << clientName << "' did not complete, now resyncing." << endl;
                }
                save();
                Client::Context ctx(ns);
                nClonedThisPass++;
                resync(ctx.db()->name);
                addDbNextPass.erase(clientName);
                incompleteCloneDbs.erase( clientName );
            }
            save();
        } else {
            bool mod;
            if ( replPair && replPair->state == ReplPair::State_Master ) {
                BSONObj id = idForOp( op, mod );
                if ( !idTracker.haveId( ns, id ) ) {
                    applyOperation( op );    
                } else if ( idTracker.haveModId( ns, id ) ) {
                    log( 6 ) << "skipping operation matching mod id object " << op << endl;
                    BSONObj existing;
                    if ( Helpers::findOne( ns, id, existing ) )
                        logOp( "i", ns, existing );
                } else {
                    log( 6 ) << "skipping operation matching changed id object " << op << endl;
                }
            } else {
                applyOperation( op );
            }
            addDbNextPass.erase( clientName );
        }
    }