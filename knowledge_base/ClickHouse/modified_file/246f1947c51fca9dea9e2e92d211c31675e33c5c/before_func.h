
    void deserialize(AggregateDataPtr place, ReadBuffer & buf, Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(ConstAggregateDataPtr place, IColumn & to) const override
    {
        auto & data_to = static_cast<ColumnArray &>(to).getData();
        auto & offsets_to = static_cast<ColumnArray &>(to).getOffsets();

        const bool first_flag = this->data(place).events.test(0);
        data_to.insert(first_flag ? Field(static_cast<UInt64>(1)) : Field(static_cast<UInt64>(0)));
        for (const auto i : ext::range(1, events_size))
        {
            if (first_flag && this->data(place).events.test(i))
