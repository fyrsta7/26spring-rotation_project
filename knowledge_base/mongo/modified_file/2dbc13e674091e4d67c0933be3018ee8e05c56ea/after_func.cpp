        //typedef list< Data > InMemory;
        typedef map< BSONObj,list<BSONObj>,BSONObjCmp > InMemory;
        typedef map< BSONObj,int,BSONObjCmp > KeyNums;

        
        class MyCmp {
        public:
            MyCmp(){}
            bool operator()( const Data &l, const Data &r ) const {
                return l.first.woCompare( r.first ) < 0;
            }
        };
    
        BSONObj reduceValues( list<BSONObj>& values , Scope * s , ScriptingFunction reduce ){
            uassert( "need values" , values.size() );
            
            int sizeEstimate = ( values.size() * values.begin()->getField( "value" ).size() ) + 128;
            BSONObj key;

            BSONObjBuilder reduceArgs( sizeEstimate );
        
            BSONObjBuilder valueBuilder( sizeEstimate );
            int n = 0;
            for ( list<BSONObj>::iterator i=values.begin(); i!=values.end(); i++){
                BSONObj o = *i;
                if ( n == 0 ){
                    reduceArgs.append( o["_id"] );
                    BSONObjBuilder temp;
                    temp.append( o["_id"] );
                    key = temp.obj();
                }
                valueBuilder.appendAs( o["value"] , BSONObjBuilder::numStr( n++ ).c_str() );
            }
        
