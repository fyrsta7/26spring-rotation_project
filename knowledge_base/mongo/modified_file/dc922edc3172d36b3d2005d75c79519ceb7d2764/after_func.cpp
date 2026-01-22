                    _scope->invoke(_reduceAll, 0, 0, 0, true);
                    return;
                }
            }

            if (_jsMode)
                return;

            bool dump = _onDisk && _size > _config.maxInMemSize;
            // attempt to reduce in memory map, if we've seen duplicates
            if ( dump || _dupCount > (_temp->size() * _config.reduceTriggerRatio)) {
				long before = _size;
				reduceInMemory();
				log(1) << "  mr: did reduceInMemory  " << before << " -->> " << _size << endl;
            }

            // reevaluate size and potentially dump
            if ( dump &&  _size > _config.maxInMemSize) {
                dumpToInc();
                log(1) << "  mr: dumping to db" << endl;
            }
        }

//        boost::thread_specific_ptr<State*> _tl;

        /**
         * emit that will be called by js function
         */
        BSONObj fast_emit( const BSONObj& args, void* data ) {
            uassert( 10077 , "fast_emit takes 2 args" , args.nFields() == 2 );
            uassert( 13069 , "an emit can't be more than half max bson size" , args.objsize() < ( BSONObjMaxUserSize / 2 ) );
            
            State* state = (State*) data;
            if ( args.firstElement().type() == Undefined ) {
                BSONObjBuilder b( args.objsize() );
                b.appendNull( "" );
                BSONObjIterator i( args );
                i.next();
                b.append( i.next() );
                state->emit( b.obj() );
            }
            else {
                state->emit( args );
            }
            return BSONObj();
        }

        /**
         * function is called when we realize we cant use js mode for m/r on the 1st key
         */
        BSONObj _bailFromJS( const BSONObj& args, void* data ) {
            State* state = (State*) data;
            state->bailFromJS();

            // emit this particular key if there is one
            if (!args.isEmpty()) {
                fast_emit(args, data);
            }
            return BSONObj();
        }

        /**
         * This class represents a map/reduce command executed on a single server
         */
        class MapReduceCommand : public Command {
        public:
            MapReduceCommand() : Command("mapReduce", false, "mapreduce") {}
            virtual bool slaveOk() const { return !replSet; }
            virtual bool slaveOverrideOk() { return true; }

            virtual void help( stringstream &help ) const {
                help << "Run a map/reduce operation on the server.\n";
                help << "Note this is used for aggregation, not querying, in MongoDB.\n";
                help << "http://www.mongodb.org/display/DOCS/MapReduce";
            }
            virtual LockType locktype() const { return NONE; }
            bool run(const string& dbname , BSONObj& cmd, string& errmsg, BSONObjBuilder& result, bool fromRepl ) {
                Timer t;
                Client::GodScope cg;
                Client& client = cc();
                CurOp * op = client.curop();

                Config config( dbname , cmd );

                log(1) << "mr ns: " << config.ns << endl;

                bool shouldHaveData = false;

                long long num = 0;
                long long inReduce = 0;

                BSONObjBuilder countsBuilder;
                BSONObjBuilder timingBuilder;
                State state( config );
                if ( ! state.sourceExists() ) {
                    errmsg = "ns doesn't exist";
                    return false;
                }

                if (replSet && state.isOnDisk()) {
                    // this means that it will be doing a write operation, make sure we are on Master
                    // ideally this check should be in slaveOk(), but at that point config is not known
                    if (!isMaster(dbname.c_str())) {
                        errmsg = "not master";
                        return false;
                    }
                }

                try {
                    state.init();

                    {
                        State** s = new State*();
                        s[0] = &state;
//                        _tl.reset( s );
                    }

                    wassert( config.limit < 0x4000000 ); // see case on next line to 32 bit unsigned
                    ProgressMeterHolder pm( op->setMessage( "m/r: (1/3) emit phase" , state.incomingDocuments() ) );
                    long long mapTime = 0;
                    {
                        readlock lock( config.ns );
                        Client::Context ctx( config.ns );

                        ShardChunkManagerPtr chunkManager;
                        if ( shardingState.needShardChunkManager( config.ns ) ) {
                            chunkManager = shardingState.getShardChunkManager( config.ns );
                        }

                        // obtain cursor on data to apply mr to, sorted
                        shared_ptr<Cursor> temp = newQueryOptimizerCursor( config.ns.c_str(), config.filter, config.sort );
                        auto_ptr<ClientCursor> cursor( new ClientCursor( QueryOption_NoCursorTimeout , temp , config.ns.c_str() ) );

                        Timer mt;
                        // go through each doc
                        while ( cursor->ok() ) {
                            if ( ! cursor->currentMatches() ) {
                                cursor->advance();
                                continue;
                            }

                            // make sure we dont process duplicates in case data gets moved around during map
                            // TODO This won't actually help when data gets moved, it's to handle multikeys.
                            if ( cursor->currentIsDup() ) {
                                cursor->advance();
                                continue;
                            }
                                                        
                            BSONObj o = cursor->current();
                            cursor->advance();

                            // check to see if this is a new object we don't own yet
                            // because of a chunk migration
                            if ( chunkManager && ! chunkManager->belongsToMe( o ) )
                                continue;

                            // do map
                            if ( config.verbose ) mt.reset();
                            config.mapper->map( o );
                            if ( config.verbose ) mapTime += mt.micros();

                            num++;
                            if ( num % 1000 == 0 ) {
                                // try to yield lock regularly
                                ClientCursor::YieldLock yield (cursor.get());
                                Timer t;
                                // check if map needs to be dumped to disk
                                state.checkSize();
                                inReduce += t.micros();

                                if ( ! yield.stillOk() ) {
                                    cursor.release();
                                    break;
                                }

                                killCurrentOp.checkForInterrupt();
                            }
                            pm.hit();

                            if ( config.limit && num >= config.limit )
                                break;
                        }
                    }
                    pm.finished();

