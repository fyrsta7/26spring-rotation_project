
    void append_escaped_for_json(const StringView&);
    void append_bytes(ReadonlyBytes);

    template<typename... Parameters>
    void appendff(CheckedFormatString<Parameters...>&& fmtstr, const Parameters&... parameters)
    {
        // FIXME: This is really not the way to go about it, but vformat expects a
        //        StringBuilder. Why does this class exist anyways?
