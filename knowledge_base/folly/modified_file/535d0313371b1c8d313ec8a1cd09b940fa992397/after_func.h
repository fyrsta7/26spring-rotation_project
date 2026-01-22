  return a;
}

// When a and b are equivalent objects, we return a to
// make sorting stable.
template <typename T, typename... Ts>
constexpr T constexpr_min(T a, Ts... ts) {
