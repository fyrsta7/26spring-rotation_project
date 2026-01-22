        BufBuilder &builder() { return b_; }
        bool scanAndOrderRequired() const { return ordering_; }
        auto_ptr< Cursor > cursor() { return c_; }
        auto_ptr< CoveredIndexMatcher > matcher() { return matcher_; }
        int n() const { return n_; }
        long long nscanned() const { return nscanned_; }
        bool saveClientCursor() const { return saveClientCursor_; }
        bool mayCreateCursor2() const { return ( queryOptions_ & QueryOption_CursorTailable ) && ntoreturn_ != 1; }
    private:
        BufBuilder b_;
        int ntoskip_;
        int ntoreturn_;
        BSONObj order_;
        bool wantMore_;
        bool explain_;
        FieldMatcher *filter_;   
        bool ordering_;
        auto_ptr< Cursor > c_;
        long long nscanned_;
        int queryOptions_;
        auto_ptr< CoveredIndexMatcher > matcher_;
        int n_;
        int soSize_;
        bool saveClientCursor_;
        auto_ptr< ScanAndOrder > so_;
        bool findingStart_;
        ClientCursor * findingStartCursor_;
        Timer findingStartTimer_;
        FindingStartMode findingStartMode_;
    };
    
    /* run a query -- includes checking for and running a Command */
    auto_ptr< QueryResult > runQuery(Message& m, QueryMessage& q, CurOp& curop ) {
        StringBuilder& ss = curop.debug().str;
        ParsedQuery pq( q );
        const char *ns = q.ns;
        int ntoskip = q.ntoskip;
        BSONObj jsobj = q.query;
        int queryOptions = q.queryOptions;
        BSONObj snapshotHint;
        
        if( logLevel >= 2 )
            log() << "runQuery: " << ns << jsobj << endl;
        
        long long nscanned = 0;
        ss << "query " << ns << " ntoreturn:" << pq.getNumToReturn();
        curop.setQuery(jsobj);
        
        BSONObjBuilder cmdResBuf;
        long long cursorid = 0;
        
        auto_ptr< QueryResult > qr;
        int n = 0;
        
        Client& c = cc();

        if ( pq.couldBeCommand() ){
            BufBuilder bb;
            bb.skip(sizeof(QueryResult));

            if ( runCommands(ns, jsobj, curop, bb, cmdResBuf, false, queryOptions) ) {
                ss << " command ";
                curop.markCommand();
                n = 1;
                qr.reset( (QueryResult *) bb.buf() );
                bb.decouple();
                qr->setResultFlagsToOk();
                qr->len = bb.len();
                ss << " reslen:" << bb.len();
                //	qr->channel = 0;
                qr->setOperation(opReply);
                qr->cursorId = cursorid;
                qr->startingFrom = 0;
                qr->nReturned = n;
            }
            return qr;
        }
        
        // regular query

        mongolock lk(false); // read lock
        Client::Context ctx( ns , dbpath , &lk );

        /* we allow queries to SimpleSlave's -- but not to the slave (nonmaster) member of a replica pair 
           so that queries to a pair are realtime consistent as much as possible.  use setSlaveOk() to 
           query the nonmaster member of a replica pair.
        */
        uassert( 10107 , "not master" , isMaster() || pq.hasOption( QueryOption_SlaveOk ) || replSettings.slave == SimpleSlave );

        BSONElement hint = useHints ? pq.getHint() : BSONElement();
        BSONObj min = pq.getMin();
        BSONObj max = pq.getMax();
        bool explain = pq.isExplain();
        bool snapshot = pq.isSnapshot();
        BSONObj query = pq.getFilter();
        BSONObj order = pq.getOrder();

        if( snapshot ) { 
            NamespaceDetails *d = nsdetails(ns);
            if ( d ){
                int i = d->findIdIndex();
                if( i < 0 ) { 
                    if ( strstr( ns , ".system." ) == 0 )
                        log() << "warning: no _id index on $snapshot query, ns:" << ns << endl;
                }
                else {
                    /* [dm] the name of an _id index tends to vary, so we build the hint the hard way here.
                       probably need a better way to specify "use the _id index" as a hint.  if someone is
                       in the query optimizer please fix this then!
                    */
                    BSONObjBuilder b;
                    b.append("$hint", d->idx(i).indexName());
                    snapshotHint = b.obj();
                    hint = snapshotHint.firstElement();
                }
            }
        }
            
        /* The ElemIter will not be happy if this isn't really an object. So throw exception
           here when that is true.
           (Which may indicate bad data from client.)
        */
        if ( query.objsize() == 0 ) {
            out() << "Bad query object?\n  jsobj:";
            out() << jsobj.toString() << "\n  query:";
            out() << query.toString() << endl;
            uassert( 10110 , "bad query object", false);
        }
            

        if ( isSimpleIdQuery( query ) ){
            nscanned = 1;

            bool nsFound = false;
            bool indexFound = false;

            BSONObj resObject;
            bool found = Helpers::findById( c, ns , query , resObject , &nsFound , &indexFound );
            if ( nsFound == false || indexFound == true ){
                BufBuilder bb(sizeof(QueryResult)+resObject.objsize()+32);
                bb.skip(sizeof(QueryResult));
                
                ss << " idhack ";
                if ( found ){
                    n = 1;
                    fillQueryResultFromObj( bb , pq.getFields() , resObject );
                }
                qr.reset( (QueryResult *) bb.buf() );
                bb.decouple();
                qr->setResultFlagsToOk();
                qr->len = bb.len();
                ss << " reslen:" << bb.len();
                qr->setOperation(opReply);
                qr->cursorId = cursorid;
                qr->startingFrom = 0;
                qr->nReturned = n;       
                return qr;
            }     
        }
        
        // regular, not QO bypass query
        
        BSONObj oldPlan;
        if ( explain && hint.eoo() && min.isEmpty() && max.isEmpty() ) {
            QueryPlanSet qps( ns, query, order );
            if ( qps.usingPrerecordedPlan() )
                oldPlan = qps.explain();
        }
        QueryPlanSet qps( ns, query, order, &hint, !explain, min, max );
        UserQueryOp original( ntoskip, pq.getNumToReturn(), order, pq.wantMore(), explain, pq.getFields() , queryOptions );
        shared_ptr< UserQueryOp > o = qps.runOp( original );
        UserQueryOp &dqo = *o;
        massert( 10362 ,  dqo.exceptionMessage(), dqo.complete() );
        n = dqo.n();
        nscanned = dqo.nscanned();
        if ( dqo.scanAndOrderRequired() )
            ss << " scanAndOrder ";
        auto_ptr<Cursor> cursor = dqo.cursor();
        log( 5 ) << "   used cursor: " << cursor.get() << endl;
        if ( dqo.saveClientCursor() ) {
            // the clientcursor now owns the Cursor* and 'c' is released:
            ClientCursor *cc = new ClientCursor(cursor, ns, !(queryOptions & QueryOption_NoCursorTimeout));
            cursorid = cc->cursorid;
            cc->query = jsobj.getOwned();
            DEV out() << "  query has more, cursorid: " << cursorid << endl;
            cc->matcher = dqo.matcher();
            cc->pos = n;
            cc->fields = pq.getFieldPtr();
            cc->originalMessage = m;
            cc->updateLocation();
            if ( !cc->c->ok() && cc->c->tailable() ) {
                DEV out() << "  query has no more but tailable, cursorid: " << cursorid << endl;
            } else {
                DEV out() << "  query has more, cursorid: " << cursorid << endl;
            }
        }
        if ( explain ) {
