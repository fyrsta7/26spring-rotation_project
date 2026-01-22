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
