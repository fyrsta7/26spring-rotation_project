    void Task::update(u64 value) {
        this->m_currValue = value;

        if (this->m_shouldInterrupt) [[unlikely]]
            throw TaskInterruptor();
    }