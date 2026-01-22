
  explicit F14ItemIter(Packed const& packed)
      : itemPtr_{packed.ptr()}, index_{packed.index()} {}

  F14ItemIter(ChunkPtr chunk, std::size_t index)
      : itemPtr_{std::pointer_traits<ItemPtr>::pointer_to(chunk->item(index))},
        index_{index} {
    FOLLY_SAFE_DCHECK(index < Chunk::kCapacity, "");
    folly::assume(
        std::pointer_traits<ItemPtr>::pointer_to(chunk->item(index)) !=
        nullptr);
    folly::assume(itemPtr_ != nullptr);
  }

  FOLLY_ALWAYS_INLINE void advance() {
    auto c = chunk();

    // common case is packed entries
    while (index_ > 0) {
      --index_;
      --itemPtr_;
      if (LIKELY(c->occupied(index_))) {
        return;
      }
    }

    // It's fairly common for an iterator to be advanced and then become
    // dead, for example in the return value from erase(iter) or in
    // the last step of a loop.  We'd like to make sure that the entire
    // advance() method can be eliminated by the compiler's dead code
    // elimination pass.  To do that it must eliminate the loops, which
    // requires it to prove that they have no side effects.  It's easy
    // to show that there are no escaping stores, but at the moment
    // compilers also consider an infinite loop to be a side effect.
    // (There are parts of the standard that would allow them to treat
    // this as undefined behavior, but at the moment they don't exploit
    // those clauses.)
    //
    // The following loop should really be a while loop, which would
    // save a register, some instructions, and a conditional branch,
    // but by writing it as a for loop the compiler can prove to itself
    // that it will eventually terminate.  (No matter that even if the
    // loop executed in a single cycle it would take about 200 years to
    // run all 2^64 iterations.)
    //
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=82776 has the bug we
    // filed about the issue.  while (true) {
    for (std::size_t i = 1; i != 0; ++i) {
      // exhausted the current chunk
      if (UNLIKELY(c->eof())) {
