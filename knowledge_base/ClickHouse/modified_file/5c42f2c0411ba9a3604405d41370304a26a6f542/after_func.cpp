              */
            const auto & saved_column = saved_block_sample.getByPosition(right_indexes[j]).column;
            if (columns[j]->isNullable() && !saved_column->isNullable())
                nullable_column_ptrs[j] = typeid_cast<ColumnNullable *>(columns[j].get());
        }
    }

    size_t size() const { return columns.size(); }

    void buildOutput()
    {
        for (size_t i = 0; i < this->size(); ++i)
        {
            auto& col = columns[i];
            size_t default_count = 0;
            for (size_t j = 0; j < lazy_output.blocks.size(); ++j)
            {
                if (!lazy_output.blocks[j])
                {
//                    type_name[i].type->insertDefaultInto(*col);
                    default_count ++;
                    continue;
                }
                if (default_count > 0)
                {
                    JoinCommon::addDefaultValues(*col, type_name[i].type, default_count);
                    default_count = 0;
                }
                const auto & column_from_block = reinterpret_cast<const Block *>(lazy_output.blocks[j])->getByPosition(right_indexes[i]);
                /// If it's joinGetOrNull, we need to wrap not-nullable columns in StorageJoin.
                if (is_join_get)
                {
                    if (auto * nullable_col = typeid_cast<ColumnNullable *>(col.get());
                        nullable_col && !column_from_block.column->isNullable())
                    {
                        nullable_col->insertFromNotNullable(*column_from_block.column, lazy_output.row_nums[j]);
                        continue;
                    }
                }
