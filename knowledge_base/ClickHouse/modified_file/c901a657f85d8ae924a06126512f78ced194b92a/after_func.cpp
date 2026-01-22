void PrettyBlockOutputFormat::calculateWidths(
    const Block & header, const Chunk & chunk,
    WidthsPerColumn & widths, Widths & max_padded_widths, Widths & name_widths)
{
    size_t num_rows = std::min(chunk.getNumRows(), format_settings.pretty.max_rows);
    size_t num_columns = chunk.getNumColumns();
    const auto & columns = chunk.getColumns();

    widths.resize(num_columns);
    max_padded_widths.resize_fill(num_columns);
    name_widths.resize(num_columns);

    /// Calculate widths of all values.
    String serialized_value;
    size_t prefix = 2; // Tab character adjustment
    for (size_t i = 0; i < num_columns; ++i)
    {
        const auto & elem = header.getByPosition(i);
        const auto & column = columns[i];

        widths[i].resize(num_rows);

        for (size_t j = 0; j < num_rows; ++j)
        {
            {
                WriteBufferFromString out_serialize(serialized_value);
                elem.type->serializeAsText(*column, j, out_serialize, format_settings);
            }

            /// Avoid calculating width of too long strings by limiting the size in bytes.
            /// Note that it is just an estimation. 4 is the maximum size of Unicode code point in bytes in UTF-8.
            /// But it's possible that the string is long in bytes but very short in visible size.
            /// (e.g. non-printable characters, diacritics, combining characters)
            if (serialized_value.size() > format_settings.pretty.max_value_width * 4)
                serialized_value.resize(format_settings.pretty.max_value_width * 4);

            widths[i][j] = UTF8::computeWidth(reinterpret_cast<const UInt8 *>(serialized_value.data()), serialized_value.size(), prefix);
            max_padded_widths[i] = std::max(max_padded_widths[i],
                std::min(format_settings.pretty.max_column_pad_width,
                    std::min<UInt64>(format_settings.pretty.max_value_width, widths[i][j])));
        }

        /// And also calculate widths for names of columns.
        {
            // name string doesn't contain Tab, no need to pass `prefix`
            name_widths[i] = std::min<UInt64>(format_settings.pretty.max_column_pad_width,
                UTF8::computeWidth(reinterpret_cast<const UInt8 *>(elem.name.data()), elem.name.size()));
            max_padded_widths[i] = std::max(max_padded_widths[i], name_widths[i]);
        }
        prefix += max_padded_widths[i] + 3;
    }
}
