  /*
   * Determine the size type
   */
  typedef typename mpl::filter_view<
      PolicyList,
      std::is_integral<mpl::placeholders::_1>>::type Integrals;
  typedef typename mpl::eval_if<
      mpl::empty<Integrals>,
      mpl::identity<std::size_t>,
      mpl::front<Integrals>>::type SizeType;

  static_assert(
      std::is_unsigned<SizeType>::value,
      "Size type should be an unsigned integral type");
  static_assert(
      mpl::size<Integrals>::value == 0 || mpl::size<Integrals>::value == 1,
      "Multiple size types specified in small_vector<>");

  /*
   * Determine whether we should allow spilling to the heap or not.
   */
  typedef typename mpl::count<PolicyList, small_vector_policy::NoHeap>::type
      HasNoHeap;

  static_assert(
      HasNoHeap::value == 0 || HasNoHeap::value == 1,
      "Multiple copies of small_vector_policy::NoHeap "
      "supplied; this is probably a mistake");

  /*
   * Make the real policy base classes.
   */
  typedef IntegralSizePolicy<SizeType, !HasNoHeap::value> ActualSizePolicy;

  /*
   * Now inherit from them all.  This is done in such a convoluted
   * way to make sure we get the empty base optimizaton on all these
   * types to keep sizeof(small_vector<>) minimal.
   */
  typedef boost::totally_ordered1<
      small_vector<Value, RequestedMaxInline, InPolicyA, InPolicyB, InPolicyC>,
      ActualSizePolicy>
      type;
};

template <class T>
T* pointerFlagSet(T* p) {
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(p) | 1);
}
template <class T>
bool pointerFlagGet(T* p) {
  return reinterpret_cast<uintptr_t>(p) & 1;
}
template <class T>
T* pointerFlagClear(T* p) {
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t(1));
}
inline void* shiftPointer(void* p, size_t sizeBytes) {
  return static_cast<char*>(p) + sizeBytes;
}
} // namespace detail

//////////////////////////////////////////////////////////////////////
FOLLY_SV_PACK_PUSH
template <
    class Value,
    std::size_t RequestedMaxInline = 1,
    class PolicyA = void,
    class PolicyB = void,
    class PolicyC = void>
class small_vector : public detail::small_vector_base<
                         Value,
                         RequestedMaxInline,
                         PolicyA,
                         PolicyB,
                         PolicyC>::type {
  typedef typename detail::
      small_vector_base<Value, RequestedMaxInline, PolicyA, PolicyB, PolicyC>::
          type BaseType;
  typedef typename BaseType::InternalSizeType InternalSizeType;

  /*
   * Figure out the max number of elements we should inline.  (If
   * the user asks for less inlined elements than we can fit unioned
   * into our value_type*, we will inline more than they asked.)
   */
  static constexpr std::size_t MaxInline{
      constexpr_max(sizeof(Value*) / sizeof(Value), RequestedMaxInline)};

 public:
  typedef std::size_t size_type;
  typedef Value value_type;
  typedef std::allocator<Value> allocator_type;
  typedef value_type& reference;
  typedef value_type const& const_reference;
  typedef value_type* iterator;
  typedef value_type* pointer;
  typedef value_type const* const_iterator;
  typedef value_type const* const_pointer;
  typedef std::ptrdiff_t difference_type;

  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  small_vector() = default;
  // Allocator is unused here. It is taken in for compatibility with std::vector
  // interface, but it will be ignored.
  small_vector(const std::allocator<Value>&) {}

  small_vector(small_vector const& o) {
    auto n = o.size();
    makeSize(n);
    {
      auto rollback = makeGuard([&] {
        if (this->isExtern()) {
          u.freeHeap();
        }
      });
      std::uninitialized_copy(o.begin(), o.end(), begin());
      rollback.dismiss();
    }
    this->setSize(n);
  }

  small_vector(small_vector&& o) noexcept(
      std::is_nothrow_move_constructible<Value>::value) {
    if (o.isExtern()) {
      swap(o);
    } else {
      std::uninitialized_copy(
          std::make_move_iterator(o.begin()),
          std::make_move_iterator(o.end()),
          begin());
      this->setSize(o.size());
    }
  }

  small_vector(std::initializer_list<value_type> il) {
    constructImpl(il.begin(), il.end(), std::false_type());
  }

  explicit small_vector(size_type n) {
    doConstruct(n, [&](void* p) { new (p) value_type(); });
  }

  small_vector(size_type n, value_type const& t) {
    doConstruct(n, [&](void* p) { new (p) value_type(t); });
  }

  template <class Arg>
  explicit small_vector(Arg arg1, Arg arg2) {
    // Forward using std::is_arithmetic to get to the proper
    // implementation; this disambiguates between the iterators and
    // (size_t, value_type) meaning for this constructor.
    constructImpl(arg1, arg2, std::is_arithmetic<Arg>());
  }

  ~small_vector() {
    for (auto& t : *this) {
      (&t)->~value_type();
    }
    if (this->isExtern()) {
      u.freeHeap();
    }
  }

  small_vector& operator=(small_vector const& o) {
    if (FOLLY_LIKELY(this != &o)) {
      assign(o.begin(), o.end());
    }
    return *this;
  }

  small_vector& operator=(small_vector&& o) {
    // TODO: optimization:
    // if both are internal, use move assignment where possible
    if (FOLLY_LIKELY(this != &o)) {
      clear();
      swap(o);
    }
    return *this;
  }

  bool operator==(small_vector const& o) const {
    return size() == o.size() && std::equal(begin(), end(), o.begin());
  }

  bool operator<(small_vector const& o) const {
    return std::lexicographical_compare(begin(), end(), o.begin(), o.end());
  }

  static constexpr size_type max_size() {
    return !BaseType::kShouldUseHeap ? static_cast<size_type>(MaxInline)
                                     : BaseType::policyMaxSize();
  }

  allocator_type get_allocator() const {
    return {};
  }

  size_type size() const {
    return this->doSize();
  }
  bool empty() const {
    return !size();
  }

  iterator begin() {
    return data();
  }
  iterator end() {
    return data() + size();
  }
  const_iterator begin() const {
    return data();
  }
  const_iterator end() const {
    return data() + size();
  }
  const_iterator cbegin() const {
    return begin();
  }
  const_iterator cend() const {
    return end();
  }

  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }
  reverse_iterator rend() {
    return reverse_iterator(begin());
  }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  const_reverse_iterator crbegin() const {
    return rbegin();
  }
  const_reverse_iterator crend() const {
    return rend();
  }

  /*
   * Usually one of the simplest functions in a Container-like class
   * but a bit more complex here.  We have to handle all combinations
   * of in-place vs. heap between this and o.
   *
   * Basic guarantee only.  Provides the nothrow guarantee iff our
   * value_type has a nothrow move or copy constructor.
   */
  void swap(small_vector& o) {
    using std::swap; // Allow ADL on swap for our value_type.

    if (this->isExtern() && o.isExtern()) {
      this->swapSizePolicy(o);

      auto thisCapacity = this->capacity();
      auto oCapacity = o.capacity();

      auto* tmp = u.pdata_.heap_;
      u.pdata_.heap_ = o.u.pdata_.heap_;
      o.u.pdata_.heap_ = tmp;

      this->setCapacity(oCapacity);
      o.setCapacity(thisCapacity);

      return;
    }

    if (!this->isExtern() && !o.isExtern()) {
      auto& oldSmall = size() < o.size() ? *this : o;
      auto& oldLarge = size() < o.size() ? o : *this;

      for (size_type i = 0; i < oldSmall.size(); ++i) {
        swap(oldSmall[i], oldLarge[i]);
      }

      size_type i = oldSmall.size();
      const size_type ci = i;
      {
        auto rollback = makeGuard([&] {
          oldSmall.setSize(i);
          for (; i < oldLarge.size(); ++i) {
            oldLarge[i].~value_type();
          }
          oldLarge.setSize(ci);
        });
        for (; i < oldLarge.size(); ++i) {
          auto addr = oldSmall.begin() + i;
          new (addr) value_type(std::move(oldLarge[i]));
          oldLarge[i].~value_type();
        }
        rollback.dismiss();
      }
      oldSmall.setSize(i);
      oldLarge.setSize(ci);
      return;
    }

    // isExtern != o.isExtern()
    auto& oldExtern = o.isExtern() ? o : *this;
    auto& oldIntern = o.isExtern() ? *this : o;

    auto oldExternCapacity = oldExtern.capacity();
    auto oldExternHeap = oldExtern.u.pdata_.heap_;

    auto buff = oldExtern.u.buffer();
    size_type i = 0;
    {
      auto rollback = makeGuard([&] {
        for (size_type kill = 0; kill < i; ++kill) {
          buff[kill].~value_type();
        }
        for (; i < oldIntern.size(); ++i) {
          oldIntern[i].~value_type();
        }
        oldIntern.setSize(0);
        oldExtern.u.pdata_.heap_ = oldExternHeap;
        oldExtern.setCapacity(oldExternCapacity);
      });
      for (; i < oldIntern.size(); ++i) {
        new (&buff[i]) value_type(std::move(oldIntern[i]));
        oldIntern[i].~value_type();
      }
      rollback.dismiss();
    }
    oldIntern.u.pdata_.heap_ = oldExternHeap;
    this->swapSizePolicy(o);
    oldIntern.setCapacity(oldExternCapacity);
  }

  void resize(size_type sz) {
    if (sz < size()) {
      erase(begin() + sz, end());
      return;
    }
    makeSize(sz);
    detail::populateMemForward(
        begin() + size(), sz - size(), [&](void* p) { new (p) value_type(); });
    this->setSize(sz);
  }

  void resize(size_type sz, value_type const& v) {
    if (sz < size()) {
      erase(begin() + sz, end());
      return;
    }
    makeSize(sz);
    detail::populateMemForward(
        begin() + size(), sz - size(), [&](void* p) { new (p) value_type(v); });
    this->setSize(sz);
  }

  value_type* data() noexcept {
    return this->isExtern() ? u.heap() : u.buffer();
  }

  value_type const* data() const noexcept {
    return this->isExtern() ? u.heap() : u.buffer();
  }

  template <class... Args>
  iterator emplace(const_iterator p, Args&&... args) {
    if (p == cend()) {
      emplace_back(std::forward<Args>(args)...);
      return end() - 1;
    }

    /*
     * We implement emplace at places other than at the back with a
     * temporary for exception safety reasons.  It is possible to
     * avoid having to do this, but it becomes hard to maintain the
     * basic exception safety guarantee (unless you respond to a copy
     * constructor throwing by clearing the whole vector).
     *
     * The reason for this is that otherwise you have to destruct an
     * element before constructing this one in its place---if the
     * constructor throws, you either need a nothrow default
     * constructor or a nothrow copy/move to get something back in the
     * "gap", and the vector requirements don't guarantee we have any
     * of these.  Clearing the whole vector is a legal response in
     * this situation, but it seems like this implementation is easy
     * enough and probably better.
     */
    return insert(p, value_type(std::forward<Args>(args)...));
  }

  void reserve(size_type sz) {
    makeSize(sz);
  }

  size_type capacity() const {
    if (this->isExtern()) {
      if (u.hasCapacity()) {
        return u.getCapacity();
      }
      return malloc_usable_size(u.pdata_.heap_) / sizeof(value_type);
    }
    return MaxInline;
  }

  void shrink_to_fit() {
    if (!this->isExtern()) {
      return;
    }

    small_vector tmp(begin(), end());
    tmp.swap(*this);
  }

  template <class... Args>
  reference emplace_back(Args&&... args) {
    if (capacity() == size()) {
      // Any of args may be references into the vector.
      // When we are reallocating, we have to be careful to construct the new
      // element before modifying the data in the old buffer.
      makeSize(
          size() + 1,
          [&](void* p) { new (p) value_type(std::forward<Args>(args)...); },
          size());
    } else {
      new (end()) value_type(std::forward<Args>(args)...);
    }
    this->setSize(size() + 1);
    return back();
  }

  void push_back(value_type&& t) {
    emplace_back(std::move(t));
  }

  void push_back(value_type const& t) {
    emplace_back(t);
  }

  void pop_back() {
    erase(end() - 1);
  }

  iterator insert(const_iterator constp, value_type&& t) {
    iterator p = unconst(constp);

    if (p == end()) {
      push_back(std::move(t));
      return end() - 1;
    }

    auto offset = p - begin();

    if (capacity() == size()) {
      makeSize(
          size() + 1,
          [&t](void* ptr) { new (ptr) value_type(std::move(t)); },
          offset);
      this->setSize(this->size() + 1);
    } else {
      detail::moveObjectsRightAndCreate(
          data() + offset,
          data() + size(),
          data() + size() + 1,
          [&](size_t i) -> value_type&& {
            assert(i == 0);
            (void)i;
            return std::move(t);
          });
      this->setSize(size() + 1);
    }
    return begin() + offset;
  }

  iterator insert(const_iterator p, value_type const& t) {
    // Make a copy and forward to the rvalue value_type&& overload
    // above.
    return insert(p, value_type(t));
  }

  iterator insert(const_iterator pos, size_type n, value_type const& val) {
    auto offset = pos - begin();
    makeSize(size() + n);
    detail::moveObjectsRightAndCreate(
        data() + offset,
        data() + size(),
        data() + size() + n,
        [&](size_t i) -> value_type const& {
          assert(i < n);
          (void)i;
          return val;
        });
    this->setSize(size() + n);
    return begin() + offset;
  }

  template <class Arg>
  iterator insert(const_iterator p, Arg arg1, Arg arg2) {
    // Forward using std::is_arithmetic to get to the proper
    // implementation; this disambiguates between the iterators and
    // (size_t, value_type) meaning for this function.
    return insertImpl(unconst(p), arg1, arg2, std::is_arithmetic<Arg>());
  }

  iterator insert(const_iterator p, std::initializer_list<value_type> il) {
    return insert(p, il.begin(), il.end());
  }

  iterator erase(const_iterator q) {
    std::move(unconst(q) + 1, end(), unconst(q));
    (data() + size() - 1)->~value_type();
    this->setSize(size() - 1);
    return unconst(q);
  }

  iterator erase(const_iterator q1, const_iterator q2) {
    if (q1 == q2) {
      return unconst(q1);
    }
    std::move(unconst(q2), end(), unconst(q1));
    for (auto it = (end() - std::distance(q1, q2)); it != end(); ++it) {
      it->~value_type();
    }
    this->setSize(size() - (q2 - q1));
    return unconst(q1);
  }

  void clear() {
    // Equivalent to erase(begin(), end()), but neither Clang or GCC are able to
    // optimize away the abstraction.
    for (auto it = begin(); it != end(); ++it) {
      it->~value_type();
    }
    this->setSize(0);
  }

  template <class Arg>
  void assign(Arg first, Arg last) {
    clear();
    insert(end(), first, last);
  }

  void assign(std::initializer_list<value_type> il) {
    assign(il.begin(), il.end());
  }

  void assign(size_type n, const value_type& t) {
    clear();
    insert(end(), n, t);
  }

  reference front() {
    assert(!empty());
    return *begin();
  }
  reference back() {
    assert(!empty());
    return *(end() - 1);
  }
  const_reference front() const {
    assert(!empty());
    return *begin();
  }
  const_reference back() const {
    assert(!empty());
    return *(end() - 1);
  }

  reference operator[](size_type i) {
    assert(i < size());
    return *(begin() + i);
  }

  const_reference operator[](size_type i) const {
    assert(i < size());
    return *(begin() + i);
  }

  reference at(size_type i) {
    if (i >= size()) {
      throw_exception<std::out_of_range>("index out of range");
    }
    return (*this)[i];
  }

  const_reference at(size_type i) const {
    if (i >= size()) {
      throw_exception<std::out_of_range>("index out of range");
    }
    return (*this)[i];
  }

 private:
  static iterator unconst(const_iterator it) {
    return const_cast<iterator>(it);
  }

  // The std::false_type argument is part of disambiguating the
  // iterator insert functions from integral types (see insert().)
  template <class It>
  iterator insertImpl(iterator pos, It first, It last, std::false_type) {
    using categ = typename std::iterator_traits<It>::iterator_category;
    using it_ref = typename std::iterator_traits<It>::reference;
    if (std::is_same<categ, std::input_iterator_tag>::value) {
      auto offset = pos - begin();
      while (first != last) {
        pos = insert(pos, *first++);
        ++pos;
      }
      return begin() + offset;
    }

    auto const distance = std::distance(first, last);
    auto const offset = pos - begin();
    assert(distance >= 0);
    assert(offset >= 0);
    makeSize(size() + distance);
    detail::moveObjectsRightAndCreate(
        data() + offset,
        data() + size(),
        data() + size() + distance,
        [&](size_t i) -> it_ref {
          assert(i < size_t(distance));
          return *(first + i);
        });
    this->setSize(size() + distance);
    return begin() + offset;
  }

  iterator
  insertImpl(iterator pos, size_type n, const value_type& val, std::true_type) {
    // The true_type means this should call the size_t,value_type
    // overload.  (See insert().)
    return insert(pos, n, val);
  }

  // The std::false_type argument came from std::is_arithmetic as part
  // of disambiguating an overload (see the comment in the
  // constructor).
  template <class It>
  void constructImpl(It first, It last, std::false_type) {
    typedef typename std::iterator_traits<It>::iterator_category categ;
    if (std::is_same<categ, std::input_iterator_tag>::value) {
      // With iterators that only allow a single pass, we can't really
      // do anything sane here.
      while (first != last) {
        emplace_back(*first++);
      }
      return;
    }

    auto distance = std::distance(first, last);
    makeSize(distance);
    this->setSize(distance);
    {
      auto rollback = makeGuard([&] {
        if (this->isExtern()) {
          u.freeHeap();
        }
      });
      detail::populateMemForward(
          data(), distance, [&](void* p) { new (p) value_type(*first++); });
      rollback.dismiss();
    }
  }

  template <typename InitFunc>
  void doConstruct(size_type n, InitFunc&& func) {
    makeSize(n);
    this->setSize(n);
    {
      auto rollback = makeGuard([&] {
        if (this->isExtern()) {
          u.freeHeap();
        }
      });
      detail::populateMemForward(data(), n, std::forward<InitFunc>(func));
      rollback.dismiss();
    }
  }

  // The true_type means we should forward to the size_t,value_type
  // overload.
  void constructImpl(size_type n, value_type const& val, std::true_type) {
    doConstruct(n, [&](void* p) { new (p) value_type(val); });
  }

  /*
   * Compute the size after growth.
   */
  size_type computeNewSize() const {
    return std::min((3 * capacity()) / 2 + 1, max_size());
  }

  void makeSize(size_type newSize) {
    makeSizeInternal(newSize, false, [](void*) { assume_unreachable(); }, 0);
  }

  template <typename EmplaceFunc>
  void makeSize(size_type newSize, EmplaceFunc&& emplaceFunc, size_type pos) {
    assert(size() == capacity());
    makeSizeInternal(
        newSize, true, std::forward<EmplaceFunc>(emplaceFunc), pos);
  }

  /*
   * Ensure we have a large enough memory region to be size `newSize'.
   * Will move/copy elements if we are spilling to heap_ or needed to
   * allocate a new region, but if resized in place doesn't initialize
   * anything in the new region.  In any case doesn't change size().
   * Supports insertion of new element during reallocation by given
   * pointer to new element and position of new element.
   * NOTE: If reallocation is not needed, insert must be false,
   * because we only know how to emplace elements into new memory.
   */
  template <typename EmplaceFunc>
  void makeSizeInternal(
      size_type newSize,
      bool insert,
      EmplaceFunc&& emplaceFunc,
