    void append(Vector<T>&& other)
    {
        if (!m_impl) {
            m_impl = move(other.m_impl);
            return;
        }
        Vector<T> tmp = move(other);
        ensure_capacity(size() + tmp.size());
        for (auto&& v : tmp) {
            unchecked_append(move(v));
        }
    }
