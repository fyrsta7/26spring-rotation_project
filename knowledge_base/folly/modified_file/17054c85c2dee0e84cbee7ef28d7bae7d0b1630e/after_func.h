   */
  FOLLY_ALWAYS_INLINE bool isThreadEntryRemovedFromAllInMap(ThreadEntry* te) {
    uint32_t maxId = nextId_.load();
    for (uint32_t i = 0; i < maxId; ++i) {
      if (allId2ThreadEntrySets_[i].rlock()->contains(te)) {
        return false;
      }
    }
    return true;
  }

  // static helper method to reallocate the ThreadEntry::elements
  // returns != nullptr if the ThreadEntry::elements was reallocated
  // nullptr if the ThreadEntry::elements was just extended
  // and throws stdd:bad_alloc if memory cannot be allocated
  static ElementWrapper* reallocate(
      ThreadEntry* threadEntry, uint32_t idval, size_t& newCapacity);

  relaxed_atomic_uint32_t nextId_;
  std::vector<uint32_t> freeIds_;
  std::mutex lock_;
  mutable SharedMutex accessAllThreadsLock_;
  pthread_key_t pthreadKey_;
  ThreadEntry* (*threadEntry_)();
  bool strict_;
  // Total size of ElementWrapper arrays across all threads. This is meant
  // to surface the overhead of thread local tracking machinery since the array
  // can be sparse when there are lots of thread local variables under the same
  // tag.
  relaxed_atomic_int64_t totalElementWrappers_{0};
  // This is a map of all thread entries mapped to index i with active
  // elements[i];
  using SynchronizedThreadEntrySet = folly::Synchronized<ThreadEntrySet>;
  folly::atomic_grow_array<SynchronizedThreadEntrySet> allId2ThreadEntrySets_;
};

struct FakeUniqueInstance {
  template <template <typename...> class Z, typename... Key, typename... Mapped>
  FOLLY_ERASE constexpr explicit FakeUniqueInstance(
      tag_t<Z<Key..., Mapped...>>, tag_t<Key...>, tag_t<Mapped...>) noexcept {}
};

/*
 * Resets element from ThreadEntry::elements at index @id.
 * call set() on the element to reset it.
 * This is a templated method for when a deleter is not provided.
 */
template <class Ptr>
void ThreadEntry::resetElement(Ptr p, uint32_t id) {
  auto validThreadEntry = (p != nullptr && !removed_);
  cleanupElementAndSetThreadEntry(id, validThreadEntry);
  elements[id].set(p);
}

/*
 * Resets element from ThreadEntry::elements at index @id.
 * call set() on the element to reset it.
 * This is a templated method for when a deleter is not provided.
 */
template <class Ptr, class Deleter>
void ThreadEntry::resetElement(Ptr p, Deleter& d, uint32_t id) {
  auto validThreadEntry = (p != nullptr && !removed_);
  cleanupElementAndSetThreadEntry(id, validThreadEntry);
  elements[id].set(p, d);
}

// Held in a singleton to track our global instances.
// We have one of these per "Tag", by default one for the whole system
// (Tag=void).
//
// Creating and destroying ThreadLocalPtr objects, as well as thread exit
// for threads that use ThreadLocalPtr objects collide on a lock inside
// StaticMeta; you can specify multiple Tag types to break that lock.
template <class Tag, class AccessMode>
struct FOLLY_EXPORT StaticMeta final : StaticMetaBase {
 private:
  static constexpr bool IsTagVoid = std::is_void_v<Tag>;
  static constexpr bool IsAccessModeStrict =
      std::is_same_v<AccessMode, AccessModeStrict>;
  static_assert(!IsTagVoid || !IsAccessModeStrict);

  using UniqueInstance =
      conditional_t<IsTagVoid, FakeUniqueInstance, detail::UniqueInstance>;
  static UniqueInstance unique;

 public:
  StaticMeta()
      : StaticMetaBase(&StaticMeta::getThreadEntrySlow, IsAccessModeStrict) {
    AtFork::registerHandler(
        this,
        /*prepare*/ &StaticMeta::preFork,
        /*parent*/ &StaticMeta::onForkParent,
        /*child*/ &StaticMeta::onForkChild);
  }

  static StaticMeta<Tag, AccessMode>& instance() {
    (void)unique; // force the object not to be thrown out as unused
    // Leak it on exit, there's only one per process and we don't have to
    // worry about synchronization with exiting threads.
    return detail::createGlobal<StaticMeta<Tag, AccessMode>, void>();
  }

  FOLLY_EXPORT FOLLY_ALWAYS_INLINE static ElementWrapper& get(EntryID* ent) {
    // Eliminate as many branches and as much extra code as possible in the
    // cached fast path, leaving only one branch here and one indirection
    // below.

    ThreadEntry* te = getThreadEntry(ent);
    uint32_t id = ent->getOrInvalid();
    // Only valid index into the the elements array
    DCHECK_NE(id, kEntryIDInvalid);
    return te->elements[id];
  }

  /*
   * In order to facilitate adding/clearing ThreadEntry* to
   * StaticMetaBase::allId2ThreadEntrySets_ during ThreadLocalPtr
   * reset()/release() we need access to the ThreadEntry* directly. This allows
   * for direct interaction with StaticMetaBase::allId2ThreadEntrySets_. We keep
   * StaticMetaBase::allId2ThreadEntrySets_ updated with ThreadEntry* whenever a
   * ThreadLocal is set/released.
   */
  FOLLY_EXPORT FOLLY_ALWAYS_INLINE static ThreadEntry* getThreadEntry(
      EntryID* ent) {
    // Eliminate as many branches and as much extra code as possible in the
    // cached fast path, leaving only one branch here and one indirection below.
    uint32_t id = ent->getOrInvalid();
    static thread_local ThreadEntry* threadEntryTL{};
    ThreadEntry* threadEntryNonTL{};
    auto& threadEntry = kUseThreadLocal ? threadEntryTL : threadEntryNonTL;

    static thread_local size_t capacityTL{};
    size_t capacityNonTL{};
    auto& capacity = kUseThreadLocal ? capacityTL : capacityNonTL;

    if (FOLLY_UNLIKELY(capacity <= id)) {
      getSlowReserveAndCache(ent, threadEntry, capacity);
    }
    return threadEntry;
  }

  FOLLY_NOINLINE static void getSlowReserveAndCache(
      EntryID* ent, ThreadEntry*& threadEntry, size_t& capacity) {
