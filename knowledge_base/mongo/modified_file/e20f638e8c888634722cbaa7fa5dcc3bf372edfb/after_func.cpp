		    if( *o.getStringField("op") == 'n' )
			    return;
            log() << "replSet skipping bad op in oplog: " << o.toString() << endl;
            return;
        }

        Client::Context ctx(ns);
        ctx.getClient()->curop()->reset();

        /* todo : if this asserts, do we want to ignore or not? */
        applyOperation_inlock(o);
    }

    /* initial oplog application, during initial sync, after cloning. 
       @return false on failure.  
       this method returns an error and doesn't throw exceptions (i think).
    */
    bool ReplSetImpl::initialSyncOplogApplication(
        string hn, 
        const Member *primary,
        OpTime applyGTE,
        OpTime minValid)
    { 
        if( primary == 0 ) return false;

        OpTime ts;
        try {
            OplogReader r;
            if( !r.connect(hn) ) { 
                log() << "replSet initial sync error can't connect to " << hn << " to read " << rsoplog << rsLog;
                return false;
            }

            {
                BSONObjBuilder q;
                q.appendDate("$gte", applyGTE.asDate());
                BSONObjBuilder query;
                query.append("ts", q.done());
                BSONObj queryObj = query.done();
                r.query(rsoplog, queryObj);
            }
            assert( r.haveCursor() );

            /* we lock outside the loop to avoid the overhead of locking on every operation.  server isn't usable yet anyway! */
            writelock lk("");

            {
                if( !r.more() ) { 
                    sethbmsg("replSet initial sync error reading remote oplog");
                    log() << "replSet initial sync error remote oplog (" << rsoplog << ") on host " << hn << " is empty?" << rsLog;
                    return false;
                }
                bo op = r.next();
                OpTime t = op["ts"]._opTime();
                r.putBack(op);

                if( op.firstElement().fieldName() == string("$err") ) { 
                    log() << "replSet initial sync error querying " << rsoplog << " on " << hn << " : " << op.toString() << rsLog;
                    return false;
                }

                uassert( 13508 , str::stream() << "no 'ts' in first op in oplog: " << op , !t.isNull() );
                if( t > applyGTE ) {
                    sethbmsg(str::stream() << "error " << hn << " oplog wrapped during initial sync");
                    log() << "replSet initial sync expected first optime of " << applyGTE << rsLog;
                    log() << "replSet initial sync but received a first optime of " << t << " from " << hn << rsLog;
                    return false;
                }
            }

            // todo : use exhaust
            unsigned long long n = 0;
            while( 1 ) { 

                if( !r.more() )
                    break;
                BSONObj o = r.nextSafe(); /* note we might get "not master" at some point */
                {
                    ts = o["ts"]._opTime();

                    /* if we have become primary, we dont' want to apply things from elsewhere
                        anymore. assumePrimary is in the db lock so we are safe as long as 
                        we check after we locked above. */
					const Member *p1 = box.getPrimary();
                    if( p1 != primary || replSetForceInitialSyncFailure ) {
                        int f = replSetForceInitialSyncFailure;
                        if( f > 0 ) {
                            replSetForceInitialSyncFailure = f-1;
                            log() << "replSet test code invoked, replSetForceInitialSyncFailure" << rsLog;
                        }
                        log() << "replSet primary was:" << primary->fullName() << " now:" << 
                            (p1 != 0 ? p1->fullName() : "none") << rsLog;
                        throw DBException("primary changed",0);
                    }

                    if( ts >= applyGTE ) {
                        // optimes before we started copying need not be applied.
