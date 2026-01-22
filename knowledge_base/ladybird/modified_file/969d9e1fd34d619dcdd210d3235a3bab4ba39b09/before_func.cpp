        frame = frame->prev;
    }
#endif
    return buffer;
}

static Optional<UnrealizedSourceRange> get_source_range(ExecutionContext const* context, Vector<FlatPtr> const& native_stack)
{
    // native function
    if (!context->executable)
        return {};

    auto const* native_executable = context->executable->native_executable();
    if (!native_executable) {
        // Interpreter frame
        if (context->instruction_stream_iterator.has_value())
            return context->instruction_stream_iterator->source_range();
        return {};
    }

    // JIT frame
    for (auto address : native_stack) {
        auto range = native_executable->get_source_range(*context->executable, address);
        if (range.has_value()) {
            auto realized = range->realize();
