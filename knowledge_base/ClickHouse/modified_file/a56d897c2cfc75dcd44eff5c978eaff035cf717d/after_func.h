    void clear()
    {
        return destroyElements();
    }

    void resize(size_t new_capacity)
    {
        counter_list.reserve(new_capacity);
        alpha_map.resize(nextAlphaSize(new_capacity));
        m_capacity = new_capacity;
    }

    void insert(const TKey & key, UInt64 increment = 1, UInt64 error = 0)
    {
        // Increase weight of a key that already exists
        auto hash = counter_map.hash(key);
        auto counter = findCounter(key, hash);
        if (counter)
        {
            counter->count += increment;
            counter->error += error;
            percolate(counter);
            return;
        }
        // Key doesn't exist, but can fit in the top K
        else if (unlikely(size() < capacity()))
        {
            auto c = new Counter(arena.emplace(key), increment, error, hash);
            push(c);
            return;
        }

        auto min = counter_list.back();
        // The key doesn't exist and cannot fit in the current top K, but
        // the new key has a bigger weight and is virtually more present
        // compared to the element who is less present on the set. This part
        // of the code is useful for the function topKWeighted
        if (increment > min->count)
        {
            destroyLastElement();
            push(new Counter(arena.emplace(key), increment, error, hash));
            return;
        }

        const size_t alpha_mask = alpha_map.size() - 1;
        auto & alpha = alpha_map[hash & alpha_mask];
