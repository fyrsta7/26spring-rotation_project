    void Task::update(u64 value) {
        this->m_currValue.store(value, std::memory_order_relaxed);

        if (this->m_shouldInterrupt.load(std::memory_order_relaxed)) [[unlikely]]
            throw TaskInterruptor();
    }
