        MutableValue(const MutableValue& other): _val(other._val) {}
        explicit MutableValue(Value& val): _val(val) {}

        /// Used by MutableDocument(MutableValue)
        const RefCountable*& getDocPtr() {
            if (_val.getType() != Object)
                *this = Value(Document());

