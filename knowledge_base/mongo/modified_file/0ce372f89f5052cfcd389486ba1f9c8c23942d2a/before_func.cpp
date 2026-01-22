
            // all grid commands are designed not to lock
            virtual LockType locktype() const { return NONE; }


            // default impl uses all shards for DB
            virtual void getShards(const string& dbName , BSONObj& cmdObj, set<Shard>& shards) {
                DBConfigPtr conf = grid.getDBConfig( dbName , false );
                conf->getAllShards(shards);
            }

            virtual void aggregateResults(const vector<BSONObj>& results, BSONObjBuilder& output) {}

            // don't override
            virtual bool run(const string& dbName , BSONObj& cmdObj, int, string& errmsg, BSONObjBuilder& output, bool) {
                LOG(1) << "RunOnAllShardsCommand db: " << dbName << " cmd:" << cmdObj << endl;
                set<Shard> shards;
                getShards(dbName, cmdObj, shards);

                list< shared_ptr<Future::CommandResult> > futures;
                for ( set<Shard>::const_iterator i=shards.begin(), end=shards.end() ; i != end ; i++ ) {
                    futures.push_back( Future::spawnCommand( i->getConnString() , dbName , cmdObj, 0 ) );
                }

                vector<BSONObj> results;
                BSONObjBuilder subobj (output.subobjStart("raw"));
                BSONObjBuilder errors;
                for ( list< shared_ptr<Future::CommandResult> >::iterator i=futures.begin(); i!=futures.end(); i++ ) {
                    shared_ptr<Future::CommandResult> res = *i;
                    if ( ! res->join() ) {
                        errors.appendAs(res->result()["errmsg"], res->getServer());
                    }
                    results.push_back( res->result() );
