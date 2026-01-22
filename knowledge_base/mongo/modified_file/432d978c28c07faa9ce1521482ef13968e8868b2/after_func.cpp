        s << "{ ";
        BSONObjIterator i(*this);
        BSONElement e = i.next();
        if ( !e.eoo() )
            while ( 1 ) {
                s << e.jsonString( format );
                e = i.next();
                if ( e.eoo() )
                    break;
                s << ", ";
            }
        s << " }";
        return s.str();
    }

// todo: can be a little faster if we don't use toString() here.
    bool BSONObj::valid() const {
        try{
            BSONObjIterator it(*this);
            while( true ){
                if (! it.moreWithEOO() )
                    return false;

                // both throw exception on failure
                BSONElement e = it.next(true);
                e.validate();

                if (e.eoo()){
