    }

    template <typename T, typename S>
    static NO_INLINE void executeInstructionsColumnar(std::vector<Instruction> & instructions, size_t rows, const MutableColumnPtr & res)
    {
        PaddedPODArray<S> inserts(rows, static_cast<S>(instructions.size()));
        calculateInserts(instructions, rows, inserts);

        PaddedPODArray<T> & res_data = assert_cast<ColumnVector<T> &>(*res).getData();
        for (size_t row_i = 0; row_i < rows; ++row_i)
        {
            auto & instruction = instructions[inserts[row_i]];
            auto ref = instruction.source->getDataAt(row_i);
            res_data[row_i] = *reinterpret_cast<const T*>(ref.data);
