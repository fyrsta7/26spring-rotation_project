// z = ch_p( 'a' )[ foo ] >> ( ch_p( 'b' ) | ch_p( 'c' ) );
// However, this is not always possible.  In my implementation I've tried to
// stick to the following pattern: store fields fed to action callbacks
// temporarily as ObjectBuilder members, then append to a BSONObjBuilder once
// the parser has completely matched a nonterminal and won't backtrack.  It's
// worth noting here that this parser follows a short-circuit convention.  So,
// in the original z example on line 3, if the input was "ab", foo() would only
// be called once.
    struct JsonGrammar : public grammar< JsonGrammar > {
public:
        JsonGrammar( ObjectBuilder &_b ) : b( _b ) {}

        template < typename ScannerT >
        struct definition {
            definition( JsonGrammar const &self ) {
                object = ch_p( '{' )[ objectStart( self.b ) ] >> !members >> '}';
                members = list_p((fieldName >> ':' >> value) , ',');
                fieldName =
                    str[ fieldNameEnd( self.b ) ] |
                    singleQuoteStr[ fieldNameEnd( self.b ) ] |
                    unquotedFieldName[ unquotedFieldNameEnd( self.b ) ];
                array = ch_p( '[' )[ arrayStart( self.b ) ] >> !elements >> ']';
                elements = list_p(value, ch_p(',')[arrayNext( self.b )]);
                value =
                    str[ stringEnd( self.b ) ] |
                    number |
                    integer |
                    array[ arrayEnd( self.b ) ] |
                    lexeme_d[ str_p( "true" ) ][ trueValue( self.b ) ] |
                    lexeme_d[ str_p( "false" ) ][ falseValue( self.b ) ] |
                    lexeme_d[ str_p( "null" ) ][ nullValue( self.b ) ] |
                    singleQuoteStr[ stringEnd( self.b ) ] |
                    date[ dateEnd( self.b ) ] |
                    oid[ oidEnd( self.b ) ] |
                    bindata[ binDataEnd( self.b ) ] |
                    dbref[ dbrefEnd( self.b ) ] |
                    regex[ regexEnd( self.b ) ] |
                    object[ subobjectEnd( self.b ) ] ;
                // NOTE lexeme_d and rules don't mix well, so we have this mess.
                // NOTE We use range_p rather than cntrl_p, because the latter is locale dependent.
                str = lexeme_d[ ch_p( '"' )[ chClear( self.b ) ] >>
                                *( ( ch_p( '\\' ) >>
                                     (
                                       ch_p( 'b' )[ chE( self.b ) ] |
                                       ch_p( 'f' )[ chE( self.b ) ] |
                                       ch_p( 'n' )[ chE( self.b ) ] |
                                       ch_p( 'r' )[ chE( self.b ) ] |
                                       ch_p( 't' )[ chE( self.b ) ] |
                                       ch_p( 'v' )[ chE( self.b ) ] |
                                       ( ch_p( 'u' ) >> ( repeat_p( 4 )[ xdigit_p ][ chU( self.b ) ] ) ) |
                                       ( ~ch_p('x') & (~range_p('0','9'))[ ch( self.b ) ] ) // hex and octal aren't supported
                                     )
                                   ) |
                                   ( ~range_p( 0x00, 0x1f ) & ~ch_p( '"' ) & ( ~ch_p( '\\' ) )[ ch( self.b ) ] ) ) >> '"' ];

                singleQuoteStr = lexeme_d[ ch_p( '\'' )[ chClear( self.b ) ] >>
                                *( ( ch_p( '\\' ) >>
                                     (
                                       ch_p( 'b' )[ chE( self.b ) ] |
                                       ch_p( 'f' )[ chE( self.b ) ] |
                                       ch_p( 'n' )[ chE( self.b ) ] |
                                       ch_p( 'r' )[ chE( self.b ) ] |
                                       ch_p( 't' )[ chE( self.b ) ] |
                                       ch_p( 'v' )[ chE( self.b ) ] |
                                       ( ch_p( 'u' ) >> ( repeat_p( 4 )[ xdigit_p ][ chU( self.b ) ] ) ) |
                                       ( ~ch_p('x') & (~range_p('0','9'))[ ch( self.b ) ] ) // hex and octal aren't supported
                                     )
                                   ) |
                                   ( ~range_p( 0x00, 0x1f ) & ~ch_p( '\'' ) & ( ~ch_p( '\\' ) )[ ch( self.b ) ] ) ) >> '\'' ];

                // real_p accepts numbers with nonsignificant zero prefixes, which
                // aren't allowed in JSON.  Oh well.
                number = strict_real_p[ numberValue( self.b ) ];

                static int_parser<long long, 10,  1, numeric_limits<long long>::digits10 + 1> long_long_p;
                integer = long_long_p[ intValue(self.b) ];

                // We allow a subset of valid js identifier names here.
                unquotedFieldName = lexeme_d[ ( alpha_p | ch_p( '$' ) | ch_p( '_' ) ) >> *( ( alnum_p | ch_p( '$' ) | ch_p( '_'  )) ) ];

                dbref = dbrefS | dbrefT;
                dbrefS = ch_p( '{' ) >> "\"$ref\"" >> ':' >>
                         str[ dbrefNS( self.b ) ] >> ',' >> "\"$id\"" >> ':' >> quotedOid >> '}';
                dbrefT = str_p( "Dbref" ) >> '(' >> str[ dbrefNS( self.b ) ] >> ',' >>
                         quotedOid >> ')';

                oid = oidS | oidT;
                oidS = ch_p( '{' ) >> "\"$oid\"" >> ':' >> quotedOid >> '}';
                oidT = str_p( "ObjectId" ) >> '(' >> quotedOid >> ')';

                quotedOid = lexeme_d[ '"' >> ( repeat_p( 24 )[ xdigit_p ] )[ oidValue( self.b ) ] >> '"' ];

                bindata = ch_p( '{' ) >> "\"$binary\"" >> ':' >>
                          lexeme_d[ '"' >> ( *( range_p( 'A', 'Z' ) | range_p( 'a', 'z' ) | range_p( '0', '9' ) | ch_p( '+' ) | ch_p( '/' ) ) >> *ch_p( '=' ) )[ binDataBinary( self.b ) ] >> '"' ] >> ',' >> "\"$type\"" >> ':' >>
                          lexeme_d[ '"' >> ( repeat_p( 2 )[ xdigit_p ] )[ binDataType( self.b ) ] >> '"' ] >> '}';

                // TODO: this will need to use a signed parser at some point
                date = dateS | dateT;
                dateS = ch_p( '{' ) >> "\"$date\"" >> ':' >> uint_parser< Date_t >()[ dateValue( self.b ) ] >> '}';
                dateT = !str_p("new") >> str_p( "Date" ) >> '(' >> uint_parser< Date_t >()[ dateValue( self.b ) ] >> ')';

                regex = regexS | regexT;
                regexS = ch_p( '{' ) >> "\"$regex\"" >> ':' >> str[ regexValue( self.b ) ] >> ',' >> "\"$options\"" >> ':' >> lexeme_d[ '"' >> ( *( alpha_p ) )[ regexOptions( self.b ) ] >> '"' ] >> '}';
                // FIXME Obviously it would be nice to unify this with str.
