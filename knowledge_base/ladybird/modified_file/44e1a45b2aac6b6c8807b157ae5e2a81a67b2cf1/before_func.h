    void append(Vector<T>&& other)
    {
        Vector<T> tmp = move(other);
        ensure_capacity(size() + tmp.size());
        for (auto&& v : tmp) {
            unchecked_append(move(v));
        }
    }
