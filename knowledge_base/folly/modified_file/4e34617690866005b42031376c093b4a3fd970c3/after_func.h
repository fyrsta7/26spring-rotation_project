        }                                                                 \
      }                                                                   \
      return out;                                                         \
    }                                                                     \
  };                                                                      \
  FOLLY_INLINE_VARIABLE constexpr atomic_fetch_bit_op_native_##instr##_fn \
      atomic_fetch_bit_op_native_##instr

FOLLY_DETAIL_ATOMIC_BIT_OP_DEFINE(bts);
FOLLY_DETAIL_ATOMIC_BIT_OP_DEFINE(btr);
FOLLY_DETAIL_ATOMIC_BIT_OP_DEFINE(btc);

#undef FOLLY_DETAIL_ATOMIC_BIT_OP_DEFINE

template <typename Integer, typename Op, typename Fb>
FOLLY_ERASE bool atomic_fetch_bit_op_native_(
    std::atomic<Integer>& atomic,
    std::size_t bit,
    std::memory_order order,
    Op op,
    Fb fb) {
  constexpr auto atomic_size = sizeof(Integer);
  constexpr auto lo_size = std::size_t(2);
  constexpr auto hi_size = std::size_t(8);
  // some versions of TSAN do not properly instrument the inline assembly
  if (atomic_size > hi_size || folly::kIsSanitize) {
    return fb(atomic, bit, order);
  }
