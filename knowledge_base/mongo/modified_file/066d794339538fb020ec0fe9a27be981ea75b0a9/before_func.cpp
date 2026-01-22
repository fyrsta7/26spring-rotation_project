
                    default:
                        // fall through
                        ;
                    }
                }

                // JEntry - a basic write
                verify( lenOrOpCode && lenOrOpCode < JEntry::OpCode_Min );
                _entries->rewind(4);
                e.e = (JEntry *) _entries->skip(sizeof(JEntry));
                e.dbName = e.e->isLocalDbContext() ? "local" : _lastDbName;
                verify( e.e->len == lenOrOpCode );
                _entries->skip(e.e->len);
