        BSONFieldValue<BSONObj> gt( const T& t ) const { return query( "$gt" , t ); }
        BSONFieldValue<BSONObj> lt( const T& t ) const { return query( "$lt" , t ); }

        BSONFieldValue<BSONObj> query( const char * q , const T& t ) const;
