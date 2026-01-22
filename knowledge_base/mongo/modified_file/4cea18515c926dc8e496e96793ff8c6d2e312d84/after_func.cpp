            catch(...) {
                log() << "info preallocateIsFaster couldn't run; returning false" << endl;
            }
            try { remove(p); } catch(...) { }
            return faster;
        }
        bool preallocateIsFaster() {
            Timer t;
            bool res = false;
            if( _preallocateIsFaster() && _preallocateIsFaster() ) { 
                // maybe system is just super busy at the moment? sleep a second to let it calm down.  
                // deciding to to prealloc is a medium big decision:
                sleepsecs(1);
                res = _preallocateIsFaster();
            }
            if( t.millis() > 3000 ) 
                log() << "preallocateIsFaster check took " << t.millis()/1000.0 << " secs" << endl;
            return res;
        }

        // throws
        void preallocateFile(filesystem::path p, unsigned long long len) {
            if( exists(p) ) 
                return;
            
            log() << "preallocating a journal file " << p.string() << endl;
