      */
    using ColumnIndex = UInt64;
    using Selector = PaddedPODArray<ColumnIndex>;
    virtual std::vector<MutablePtr> scatter(ColumnIndex num_columns, const Selector & selector) const = 0;

    /// Insert data from several other columns according to source mask (used in vertical merge).
    /// For now it is a helper to de-virtualize calls to insert*() functions inside gather loop
