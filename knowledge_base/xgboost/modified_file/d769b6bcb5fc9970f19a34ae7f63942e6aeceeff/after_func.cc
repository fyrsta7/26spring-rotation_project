      }
    }
  }
}

void GHistBuilder::BuildHist(const std::vector<bst_gpair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             const std::vector<bst_uint>& feat_set,
                             GHistRow hist) {
  data_.resize(nbins_ * nthread_, GHistEntry());
  std::fill(data_.begin(), data_.end(), GHistEntry());
  stat_buf_.resize(row_indices.size());

  const int K = 8;  // loop unrolling factor
  const bst_omp_uint nthread = static_cast<bst_omp_uint>(this->nthread_);
  const bst_omp_uint nrows = row_indices.end - row_indices.begin;
  const bst_omp_uint rest = nrows % K;

  #pragma omp parallel for num_threads(nthread) schedule(static)
  for (bst_omp_uint i = 0; i < nrows - rest; i += K) {
    bst_uint rid[K];
    bst_gpair stat[K];
    for (int k = 0; k < K; ++k) {
      rid[k] = row_indices.begin[i + k];
    }
    for (int k = 0; k < K; ++k) {
      stat[k] = gpair[rid[k]];
    }
    for (int k = 0; k < K; ++k) {
      stat_buf_[i + k] = stat[k];
    }
  }
  for (bst_omp_uint i = nrows - rest; i < nrows; ++i) {
    const bst_uint rid = row_indices.begin[i];
    const bst_gpair stat = gpair[rid];
    stat_buf_[i] = stat;
  }

  #pragma omp parallel for num_threads(nthread) schedule(guided)
  for (bst_omp_uint i = 0; i < nrows - rest; i += K) {
    const bst_omp_uint tid = omp_get_thread_num();
    const size_t off = tid * nbins_;
    bst_uint rid[K];
    size_t ibegin[K];
    size_t iend[K];
    bst_gpair stat[K];
    for (int k = 0; k < K; ++k) {
      rid[k] = row_indices.begin[i + k];
    }
    for (int k = 0; k < K; ++k) {
      ibegin[k] = static_cast<size_t>(gmat.row_ptr[rid[k]]);
      iend[k] = static_cast<size_t>(gmat.row_ptr[rid[k] + 1]);
    }
    for (int k = 0; k < K; ++k) {
      stat[k] = stat_buf_[i + k];
    }
    for (int k = 0; k < K; ++k) {
      for (size_t j = ibegin[k]; j < iend[k]; ++j) {
        const size_t bin = gmat.index[j];
        data_[off + bin].Add(stat[k]);
      }
    }
  }
  for (bst_omp_uint i = nrows - rest; i < nrows; ++i) {
    const bst_uint rid = row_indices.begin[i];
    const size_t ibegin = static_cast<size_t>(gmat.row_ptr[rid]);
    const size_t iend = static_cast<size_t>(gmat.row_ptr[rid + 1]);
    const bst_gpair stat = stat_buf_[i];
    for (size_t j = ibegin; j < iend; ++j) {
      const size_t bin = gmat.index[j];
      data_[bin].Add(stat);
    }
  }

  /* reduction */
  const bst_omp_uint nbins = static_cast<bst_omp_uint>(nbins_);
  #pragma omp parallel for num_threads(nthread) schedule(static)
  for (bst_omp_uint bin_id = 0; bin_id < nbins; ++bin_id) {
