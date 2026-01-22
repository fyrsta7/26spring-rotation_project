    // Returns the value at the index in the skip list from this node onwards.
    // Will return nullptr is not found.
    const Node<T, _Compare> *at(size_t idx) const;
    // Computes index of the first occurrence of a value
    bool index(const T& value, size_t &idx, size_t level) const;
    /// Number of linked lists that this node engages in, minimum 1.
    size_t height() const { return _nodeRefs.height(); }
