                log( LL_ERROR ) << "median error (1) min: " << min << " max: " << max << " median: " << median << endl;
                errmsg = "median error 1";
                return false;
            }
            else if ( x > 0 && y > 0 ){
                log( LL_ERROR ) << "median error (2) min: " << min << " max: " << max << " median: " << median << endl;
                errmsg = "median error 2";
                return false;
            }

            return true;
        }
    } cmdMedianKey;

     class SplitVector : public Command {
     public:
        SplitVector() : Command( "splitVector" , false ){}
        virtual bool slaveOk() const { return false; }
        virtual LockType locktype() const { return READ; }
        virtual void help( stringstream &help ) const {
            help <<
                "Internal command.\n"
                "example: { splitVector : \"myLargeCollection\" , keyPattern : {x:1} , maxChunkSize : 200 }\n"
                "maxChunkSize unit in MBs\n"
                "NOTE: This command may take a while to run";
        }
        bool run(const string& dbname, BSONObj& jsobj, string& errmsg, BSONObjBuilder& result, bool fromRepl ){
            const char* ns = jsobj.getStringField( "splitVector" );
            BSONObj keyPattern = jsobj.getObjectField( "keyPattern" );

            long long maxChunkSize = 0;
            BSONElement maxSizeElem = jsobj[ "maxChunkSize" ];
            if ( ! maxSizeElem.eoo() ){
                maxChunkSize = maxSizeElem.numberLong() * 1<<20;
            } else {
                errmsg = "need to specify the desired max chunk size";
                return false;
            }
            
            Client::Context ctx( ns );

            BSONObjBuilder minBuilder;
            BSONObjBuilder maxBuilder;
            BSONForEach(key, keyPattern){
                minBuilder.appendMinKey( key.fieldName() );
                maxBuilder.appendMaxKey( key.fieldName() );
            }
            BSONObj min = minBuilder.obj();
            BSONObj max = maxBuilder.obj();

            IndexDetails *idx = cmdIndexDetailsForRange( ns , errmsg , min , max , keyPattern );
            if ( idx == NULL ){
                errmsg = "couldn't find index over splitting key";
                return false;
            }

            NamespaceDetails *d = nsdetails( ns );
            BtreeCursor c( d , d->idxNo(*idx) , *idx , min , max , false , 1 );

            // We'll use the average object size and number of object to find approximately how many keys
            // each chunk should have. We'll split a little smaller than the specificied by 'maxSize'
            // assuming a recently sharded collectio is still going to grow.

            const long long dataSize = d->datasize;
            const long long recCount = d->nrecords;
            long long keyCount = 0;
            if (( dataSize > 0 ) && ( recCount > 0 )){
                const long long avgRecSize = dataSize / recCount;
                keyCount = 90 * maxChunkSize / (100 * avgRecSize);
            }

            // We traverse the index and add the keyCount-th key to the result vector. If that key
            // appeared in the vector before, we omit it. The assumption here is that all the 
            // instances of a key value live in the same chunk.

            Timer timer;
            long long currCount = 0;
            vector<BSONObj> splitKeys;
            BSONObj currKey;
            while ( c.ok() ){ 
                currCount++;
                if ( currCount > keyCount ){
                    if ( ! currKey.isEmpty() && (currKey.woCompare( c.currKey() ) == 0 ) ) 
                         continue;

                    currKey = c.currKey();
                    splitKeys.push_back( c.prettyKey( currKey ) );
                    currCount = 0;
                }
                c.advance();
