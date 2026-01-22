
        /** @param baseBuilder construct a BSONObjBuilder using an existing BufBuilder */
        BSONObjBuilder( BufBuilder &baseBuilder ) : _b( baseBuilder ), _buf( 0 ), _offset( baseBuilder.len() ), _s( this ) , _tracker(0) , _doneCalled(false) {
            _b.skip( 4 );
        }
