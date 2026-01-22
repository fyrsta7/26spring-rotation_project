void ASTLiteral::appendColumnNameImpl(WriteBuffer & ostr) const
{
    if (use_legacy_column_name_of_tuple)
    {
        appendColumnNameImplLegacy(ostr);
        return;
    }

    /// 100 - just arbitrary value.
    constexpr auto min_elements_for_hashing = 100;

    /// Special case for very large arrays and tuples. Instead of listing all elements, will use hash of them.
    /// (Otherwise column name will be too long, that will lead to significant slowdown of expression analysis.)
    auto type = value.getType();
    if ((type == Field::Types::Array && value.get<const Array &>().size() > min_elements_for_hashing)
        || (type == Field::Types::Tuple && value.get<const Tuple &>().size() > min_elements_for_hashing))
    {
        SipHash hash;
        applyVisitor(FieldVisitorHash(hash), value);
        UInt64 low, high;
        hash.get128(low, high);

        writeCString(type == Field::Types::Array ? "__array_" : "__tuple_", ostr);
        writeText(low, ostr);
        ostr.write('_');
        writeText(high, ostr);
    }
    else
    {
        /// Special case for huge AST. The `FieldVisitorToString` becomes expensive
        /// for tons of literals.
        if (value.getType() == Field::Types::String)
        {
            writeQuoted(value.get<String>(), ostr);
        }
        else
        {
            String column_name = applyVisitor(FieldVisitorToString(), value);
            writeString(column_name, ostr);
        }
    }
}
