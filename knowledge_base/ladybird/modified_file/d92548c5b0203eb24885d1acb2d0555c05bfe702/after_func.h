        m_segments.last()->data.append(forward<U>(value));
        ++m_size;
    }

    T dequeue()
    {
        VERIFY(!is_empty());
        auto value = move(m_segments.first()->data[m_index_into_first++]);
        if (m_index_into_first == segment_size) {
            delete m_segments.take_first();
            m_index_into_first = 0;
        }
        --m_size;
        if (m_size == 0 && !m_segments.is_empty()) {
            // This is not necessary for correctness but avoids faulting in
            // all the pages for the underlying Vector in the case where
            // the caller repeatedly enqueues and then dequeues a single item.
            m_index_into_first = 0;
