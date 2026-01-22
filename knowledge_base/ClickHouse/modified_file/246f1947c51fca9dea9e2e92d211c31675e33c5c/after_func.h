
    void deserialize(AggregateDataPtr place, ReadBuffer & buf, Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(ConstAggregateDataPtr place, IColumn & to) const override
    {
        auto & data_to = static_cast<ColumnUInt8 &>(static_cast<ColumnArray &>(to).getData()).getData();
        auto & offsets_to = static_cast<ColumnArray &>(to).getOffsets();

        ColumnArray::Offset current_offset = data_to.size();
        data_to.resize(current_offset + events_size);

        const bool first_flag = this->data(place).events.test(0);
        data_to[current_offset] = first_flag;
        ++current_offset;

        for (size_t i = 1; i < events_size; ++i)
        {
