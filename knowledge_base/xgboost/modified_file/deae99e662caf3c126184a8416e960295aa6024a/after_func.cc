    return std::min(rows, kNoPrefetchSize);
  }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return Prefetch::kCacheLineSize / sizeof(T);
  }
};

constexpr size_t Prefetch::kNoPrefetchSize;

template <bool do_prefetch, typename BinIdxType, bool first_page, bool any_missing = true>
void BuildHistKernel(const std::vector<GradientPair> &gpair,
                     const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                     GHistRow hist) {
  const size_t size = row_indices.Size();
  const size_t *rid = row_indices.begin;
  auto const *pgh = reinterpret_cast<const float *>(gpair.data());
  const BinIdxType *gradient_index = gmat.index.data<BinIdxType>();

  auto const &row_ptr = gmat.row_ptr.data();
  auto base_rowid = gmat.base_rowid;
  const uint32_t *offsets = gmat.index.Offset();
  auto get_row_ptr = [&](size_t ridx) {
    return first_page ? row_ptr[ridx] : row_ptr[ridx - base_rowid];
  };
  auto get_rid = [&](size_t ridx) {
    return first_page ? ridx : (ridx - base_rowid);
  };

  const size_t n_features =
      get_row_ptr(row_indices.begin[0] + 1) - get_row_ptr(row_indices.begin[0]);
  auto hist_data = reinterpret_cast<double *>(hist.data());
  const uint32_t two{2};  // Each element from 'gpair' and 'hist' contains
                          // 2 FP values: gradient and hessian.
                          // So we need to multiply each row-index/bin-index by 2
                          // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start =
        any_missing ? get_row_ptr(rid[i]) : get_rid(rid[i]) * n_features;
    const size_t icol_end =
        any_missing ? get_row_ptr(rid[i] + 1) : icol_start + n_features;

    const size_t row_size = icol_end - icol_start;
    const size_t idx_gh = two * rid[i];

    if (do_prefetch) {
      const size_t icol_start_prefetch =
          any_missing
              ? get_row_ptr(rid[i + Prefetch::kPrefetchOffset])
              : get_rid(rid[i + Prefetch::kPrefetchOffset]) * n_features;
      const size_t icol_end_prefetch =
          any_missing ? get_row_ptr(rid[i + Prefetch::kPrefetchOffset] + 1)
                      : icol_start_prefetch + n_features;

      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_end_prefetch;
           j += Prefetch::GetPrefetchStep<uint32_t>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    const BinIdxType *gr_index_local = gradient_index + icol_start;

