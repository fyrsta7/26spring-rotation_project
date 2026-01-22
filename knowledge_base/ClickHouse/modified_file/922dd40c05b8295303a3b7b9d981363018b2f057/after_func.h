        queue.splice(queue.end(), queue, cell.queue_iterator);

        return cell.value;
    }

    bool setImpl(const Key & key, const MappedPtr & mapped, [[maybe_unused]] std::lock_guard<std::mutex> & cache_lock)
    {
        auto [it, inserted] = cells.emplace(std::piecewise_construct,
            std::forward_as_tuple(key),
            std::forward_as_tuple());

        Cell & cell = it->second;
        auto value_weight = mapped ? weight_function(*mapped) : 0;

        if (inserted)
        {
            if (!removeOverflow(value_weight))
            {
                // cannot find enough space to put in the new value
                cells.erase(it);
                return false;
            }

            try
            {
                cell.queue_iterator = queue.insert(queue.end(), key);
            }
            catch (...)
            {
                cells.erase(it);
                throw;
            }
        }
        else
        {
            if (!evict_policy.canRelease(cell.value))
                return false;
            if (value_weight > cell.size && !removeOverflow(value_weight - cell.size))
                return false;
            evict_policy.release(cell.value); // release the old value. this action is empty in default policy.
            current_size -= cell.size;
            queue.splice(queue.end(), queue, cell.queue_iterator);
        }

        cell.value = mapped;
