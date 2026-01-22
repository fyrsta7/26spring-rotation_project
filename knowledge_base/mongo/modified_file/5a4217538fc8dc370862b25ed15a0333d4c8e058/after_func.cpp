            _debug.clear();
#endif
        }

#if defined(DEBUG_WRITE_INTENT)
        void assertAlreadyDeclared(void *p, int len) {
            if( commitJob.wi()._debug[p] >= len )
                return;
            log() << "assertAlreadyDeclared fails " << (void*)p << " len:" << len << ' ' << commitJob.wi()._debug[p] << endl;
            printStackTrace();
            abort();
        }
#endif

        void Writes::_insertWriteIntent(void* p, int len) {
            WriteIntent wi(p, len);

            if (_writes.empty()) {
                _writes.insert(wi);
                return;
            }

            typedef set<WriteIntent>::const_iterator iterator; // shorter

            iterator closest = _writes.lower_bound(wi);
            // closest.end() >= wi.end()

            if ((closest != _writes.end() && closest->overlaps(wi)) || // high end
                    (closest != _writes.begin() && (--closest)->overlaps(wi))) { // low end
                if (closest->contains(wi))
                    return; // nothing to do

                // find overlapping range and merge into wi
                iterator   end(closest);
                iterator begin(closest);
                while (  end->overlaps(wi)) { wi.absorb(*end); ++end; if (end == _writes.end()) break; }  // look forwards
                while (begin->overlaps(wi)) { wi.absorb(*begin); if (begin == _writes.begin()) break; --begin; } // look backwards
                if (!begin->overlaps(wi)) ++begin; // make inclusive

                DEV { // ensure we're not deleting anything we shouldn't
                    for (iterator it(begin); it != end; ++it) {
                        assert(wi.contains(*it));
                    }
                }

                _writes.erase(begin, end);
                _writes.insert(wi);
