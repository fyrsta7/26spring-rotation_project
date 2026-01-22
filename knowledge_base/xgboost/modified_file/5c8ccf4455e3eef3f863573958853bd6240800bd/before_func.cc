  });

  builder_monitor_.Stop("UpdatePredictionCache");
  return true;
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const std::vector<GradientPair>& gpair,
                                                const DMatrix& fmat,
                                                std::vector<size_t>* row_indices) {
  const auto& info = fmat.Info();
  auto& rnd = common::GlobalRandom();
  std::vector<size_t>& row_indices_local = *row_indices;
  size_t* p_row_indices = row_indices_local.data();
#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param_.subsample);
  size_t j = 0;
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (gpair[i].GetHess() >= 0.0f && coin_flip(rnd)) {
      p_row_indices[j++] = i;
    }
  }
  /* resize row_indices to reduce memory */
  row_indices_local.resize(j);
#else
  const size_t nthread = this->nthread_;
  std::vector<size_t> row_offsets(nthread, 0);
  /* usage of mt19937_64 give 2x speed up for subsampling */
  std::vector<std::mt19937> rnds(nthread);
  /* create engine for each thread */
  for (std::mt19937& r : rnds) {
    r = rnd;
  }
  const size_t discard_size = info.num_row_ / nthread;
  #pragma omp parallel num_threads(nthread)
  {
    const size_t tid = omp_get_thread_num();
    const size_t ibegin = tid * discard_size;
    const size_t iend = (tid == (nthread - 1)) ?
                        info.num_row_ : ibegin + discard_size;
    std::bernoulli_distribution coin_flip(param_.subsample);

    rnds[tid].discard(2*discard_size * tid);
    for (size_t i = ibegin; i < iend; ++i) {
      if (gpair[i].GetHess() >= 0.0f && coin_flip(rnds[tid])) {
        p_row_indices[ibegin + row_offsets[tid]++] = i;
      }
    }
  }
  /* discard global engine */
  rnd = rnds[nthread - 1];
  size_t prefix_sum = row_offsets[0];
  for (size_t i = 1; i < nthread; ++i) {
    const size_t ibegin = i * discard_size;

    for (size_t k = 0; k < row_offsets[i]; ++k) {
      row_indices_local[prefix_sum + k] = row_indices_local[ibegin + k];
    }
    prefix_sum += row_offsets[i];
