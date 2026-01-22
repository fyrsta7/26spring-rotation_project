            uow.commit();
        }

    private:
        void doInsert() {
            invariant(!_records.empty());

            KeyString value;
            for (size_t i = 0; i < _records.size(); i++) {
                value.appendRecordId(_records[i].first);
                // When there is only one record, we can omit AllZeros TypeBits. Otherwise they need
                // to be included.
                if (!(_records[i].second.isAllZeros() && _records.size() == 1)) {
                    value.appendTypeBits(_records[i].second);
                }
            }
            
            WiredTigerItem keyItem( _keyString.getBuffer(), _keyString.getSize() );
            WiredTigerItem valueItem(value.getBuffer(), value.getSize());

            _cursor->set_key(_cursor, keyItem.Get());
            _cursor->set_value(_cursor, valueItem.Get());

            invariantWTOK(_cursor->insert(_cursor));
            invariantWTOK(_cursor->reset(_cursor));

            _records.clear();
        }

        WiredTigerIndex* _idx;
        const bool _dupsAllowed;
        BSONObj _key;
        KeyString _keyString;
        std::vector<std::pair<RecordId, KeyString::TypeBits> > _records;
    };

namespace {

    /**
     * Implements the basic WT_CURSOR functionality used by both unique and standard indexes.
     */
    class WiredTigerIndexCursorBase : public SortedDataInterface::Cursor {
    public:
        WiredTigerIndexCursorBase(const WiredTigerIndex& idx, OperationContext *txn, bool forward)
           : _txn(txn),
             _cursor(idx.uri(), idx.instanceId(), false, txn),
             _idx(idx),
             _forward(forward),
             _eof(true),
             _isKeyCurrent(false) {
        }

        virtual int getDirection() const { return _forward ? 1 : -1; }
        virtual bool isEOF() const { return _eof; }

        virtual bool pointsToSamePlaceAs(const SortedDataInterface::Cursor& genOther) const {
            const WiredTigerIndexCursorBase& other =
                checked_cast<const WiredTigerIndexCursorBase&>(genOther);

            if ( _eof && other._eof )
                return true;
            else if ( _eof || other._eof )
                return false;

            // First try WT_CURSOR equals(), as this should be cheap.
            int equal;
            invariantWTOK(_cursor.get()->equals(_cursor.get(), other._cursor.get(), &equal));
            if (!equal)
                return false;

            // WT says cursors are equal, but need to double-check that the RecordIds match.
            return getRecordId() == other.getRecordId();
        }

        bool locate(const BSONObj &key, const RecordId& loc) {
            const BSONObj finalKey = stripFieldNames(key);
            fillKey(finalKey, loc);
            bool result = _locate(loc);

            // An explicit search at the start of the range should always return false
            if (loc == RecordId::min() || loc == RecordId::max() )
                return false;
            return result;
       }

        void advanceTo(const BSONObj &keyBegin,
               int keyBeginLen,
               bool afterKey,
               const vector<const BSONElement*>& keyEnd,
               const vector<bool>& keyEndInclusive) {
            // TODO: don't go to a bson obj then to a KeyString, go straight
            BSONObj key = IndexEntryComparison::makeQueryObject(
                             keyBegin, keyBeginLen,
                             afterKey, keyEnd, keyEndInclusive, getDirection() );

            fillKey(key, RecordId());
            _locate(RecordId());
        }

        void customLocate(const BSONObj& keyBegin,
                      int keyBeginLen,
                      bool afterKey,
                      const vector<const BSONElement*>& keyEnd,
                      const vector<bool>& keyEndInclusive) {
            advanceTo(keyBegin, keyBeginLen, afterKey, keyEnd, keyEndInclusive);
        }


        BSONObj getKey() const {
            if (_isKeyCurrent && !_keyBson.isEmpty())
                return _keyBson;

            loadKeyIfNeeded();
            _keyBson = KeyString::toBson(_key.getBuffer(), _key.getSize(), _idx.ordering(),
                                         getTypeBits());

            TRACE_INDEX << " returning key: " << _keyBson;
            return _keyBson;
        }

        void savePosition() {
            _savedForCheck = _txn->recoveryUnit();

            if ( !wt_keeptxnopen() && !_eof ) {
                loadKeyIfNeeded();
                _savedLoc = getRecordId();
                _cursor.reset();
            }

            _txn = NULL;
        }

        void restorePosition( OperationContext *txn ) {
            // Update the session handle with our new operation context.
            _txn = txn;
            invariant( _savedForCheck == txn->recoveryUnit() );

            if ( !wt_keeptxnopen() && !_eof ) {
                // Ensure an active session exists, so any restored cursors will bind to it
                WiredTigerRecoveryUnit::get(txn)->getSession();

                _locate(_savedLoc);
            }
        }

    protected:
        // Uses _key for the key.
        virtual bool _locate(RecordId loc) = 0;

        // Must invalidateCache()
        virtual void fillKey(const BSONObj& key, RecordId loc) = 0;

        virtual const KeyString::TypeBits& getTypeBits() const = 0;

        void advanceWTCursor() {
            invalidateCache();
            WT_CURSOR *c = _cursor.get();
            int ret = _forward ? c->next(c) : c->prev(c);
            if ( ret == WT_NOTFOUND ) {
                _eof = true;
                return;
            }
            invariantWTOK(ret);
            _eof = false;
        }

        // Seeks to _key. Returns true on exact match.
        bool seekWTCursor() {
            invalidateCache();
            WT_CURSOR *c = _cursor.get();

            int cmp = -1;
            const WiredTigerItem keyItem(_key.getBuffer(), _key.getSize());
            c->set_key(c, keyItem.Get());

            int ret = c->search_near(c, &cmp);
            if ( ret == WT_NOTFOUND ) {
                _eof = true;
                TRACE_CURSOR << "\t not found";
                return false;
            }
            invariantWTOK( ret );
            _eof = false;

            TRACE_CURSOR << "\t cmp: " << cmp;

            if (cmp == 0) {
                // Found it! This means _key must be current. Double check in DEV mode.
                _isKeyCurrent = true;
                dassertKeyCacheIsValid();
                return true;
            }

            // Make sure we land on a matching key
            if (_forward) {
                // We need to be >=
                if (cmp < 0) {
                    ret = c->next(c);
                }
            }
            else {
                // We need to be <=
                if (cmp > 0) {
                    ret = c->prev(c);
                }
            }
