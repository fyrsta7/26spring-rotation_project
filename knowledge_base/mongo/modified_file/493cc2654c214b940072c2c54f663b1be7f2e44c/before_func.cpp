                                          ( ch_p( 'u' ) >> ( repeat_p( 4 )[ xdigit_p ][ chU( self.b ) ] ) ) ) ) |
                                      ( ~range_p( 0x00, 0x1f ) & ~ch_p( '/' ) & ( ~ch_p( '\\' ) )[ ch( self.b ) ] ) ) >> str_p( "/" )[ regexValue( self.b ) ]
                                   >> ( *( ch_p( 'i' ) | ch_p( 'g' ) | ch_p( 'm' ) ) )[ regexOptions( self.b ) ] ];
            }
            rule< ScannerT > object, members, array, elements, value, str, number, integer,
            dbref, dbrefS, dbrefT, oid, oidS, oidT, bindata, date, dateS, dateT,
            regex, regexS, regexT, quotedOid, fieldName, unquotedFieldName, singleQuoteStr;
            const rule< ScannerT > &start() const {
                return object;
            }
        };
        ObjectBuilder &b;
    };

    BSONObj fromjson( const char *str ) {
        if ( ! strlen(str) )
            return BSONObj();
        ObjectBuilder b;
