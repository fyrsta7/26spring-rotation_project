  if (mapping_scheme.GetIndexingOrder() == kStridedIndexingX) {
    return thread_id_x;
  } else if (mapping_scheme.GetIndexingOrder() == kLinearStridedIndexingX) {
    int vector_size = mapping_scheme.GetVectorSize();
    return b->CreateMul(thread_id_x,
                        llvm::ConstantInt::get(index_ty, vector_size));
  }
  int64 x_num_steps =
      mapping_scheme.GetTileSizeX() / mapping_scheme.GetNumThreadsX();
  return b->CreateMul(thread_id_x,
                      llvm::ConstantInt::get(index_ty, x_num_steps));
}

void IrEmitterUnnested::EmitTile(
    const KernelMappingScheme& mapping_scheme,
    const IrArray::Index& tile_origin_index, const string& loop_name,
    KernelSupportLibrary* ksl, const ThreadIdInfo& thread_id_info,
    llvm::Value* tile_height, llvm::Value* tile_width,
    const IrEmitterUnnested::EmitElementFunction& emit_elem_function) {
  llvm::Type* index_ty = tile_width->getType();
  auto constant = [&](int64 val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  int64 num_threads_x = mapping_scheme.GetNumThreadsX();
  llvm::Value* num_threads_y = constant(mapping_scheme.GetNumThreadsY());
  int64 tile_size_x = mapping_scheme.GetTileSizeX();

  int64 x_num_steps = tile_size_x / num_threads_x;
  llvm::Value* start_offset_x = GetStartOffsetX(
      mapping_scheme, thread_id_info.thread_id_x, index_ty, &b_);

  // Using dilated mapping scheme, each thread steps with a stride of number
  // of threads.
  // Otherwise, the stride is one, but we multiply each offset by the limit of
  // number of steps which can be made.
  int64 step_x =
      mapping_scheme.GetIndexingOrder() == kLinearIndexingX ? 1 : num_threads_x;

  IrArray::Index source_idx =
      tile_origin_index.AddOffsetToDim(start_offset_x, kDimX, &b_);

  auto ceil_of_ratio = [&](llvm::Value* a, llvm::Value* b) {
    return b_.CreateUDiv(b_.CreateAdd(b_.CreateAdd(a, b), constant(-1)), b);
  };

  // True iff all threads always execute all instructions in the tiling
  // dimension X.
  bool x_tile_fits = mapping_scheme.GetDimsInElems()[kDimX] % tile_size_x == 0;

  // The outer loop below is simply doing:
  //
  // for (int y_loc=thread_id_y; y_loc<tile_height; y_loc+=num_threads_y)
  //
  //
  // However, in order to avoid an LLVM optimization triggering the ptxas bug,
  // we write this loop in a convoluted way:
  //
  // y_bound = ceil_of_ratio(tile_height - thread_id_y, num_threads_y)
  // for (int y_indvar=0; y_indvar<y_bound; y_indvar+=1)
  //    y_loc = thread_id_y + y_indvar * num_threads_y
  //
  // TODO(cheshire): Once ptxas is fixed and TF switches to it, remove the
  // workaround.
  int vector_size = mapping_scheme.GetVectorSize();
  ksl->For(
      loop_name + "_y_in_tile",
      /*start=*/constant(0),
      /*end=*/
      ceil_of_ratio(b_.CreateSub(tile_height, thread_id_info.thread_id_y),
                    num_threads_y),
      /*step=*/constant(1), [&](llvm::Value* y_indvar) {
        llvm::Value* y_loc = b_.CreateAdd(
            thread_id_info.thread_id_y, b_.CreateMul(y_indvar, num_threads_y));
        auto unroll = [&](bool add_index_boundary_condition,
                          int64 vector_size) {
          for (int64 j = 0; j < x_num_steps / vector_size; j++) {
            // Prep some values. If we do not do this, LLVM doesn't vectorize.
            llvm::Value* x_loc_base =
                b_.CreateAdd(constant(j * step_x * vector_size), start_offset_x,
                             "x_loc_base");
            IrArray::Index source_idx_x_base =
                source_idx.AddOffsetToDim(y_loc, kDimY, &b_)
                    .AddOffsetToDim(constant(j * step_x * vector_size), kDimX,
                                    &b_);

            for (int i = 0; i < vector_size; i++) {
              int old_j = j * vector_size + i;
              llvm::Value* x_loc = b_.CreateAdd(constant(i), x_loc_base, "x_loc");
              IrArray::Index source_idx_x =
                  source_idx_x_base.AddOffsetToDim(constant(i), kDimX, &b_);
              auto emit_element = [&] {
                return emit_elem_function(source_idx_x, y_loc, x_loc, old_j);
              };
              if (add_index_boundary_condition) {
                ksl->If(loop_name + "_x_in_tile",
                        b_.CreateICmpULT(x_loc, tile_width), emit_element);
              } else {
                emit_element();
              }
            }
          }
        };

        if (!x_tile_fits &&
            mapping_scheme.GetIndexingOrder() == kLinearStridedIndexingX) {
          // Only try this path when we try to vectorize the loads.

          // Special case when the tile doesn't fit completly for even row size.
          // For odd row size every other row isn't aligned, so can't be
          // vectorized.
          ksl->If(loop_name + "_is_full_tile",
                  // if (block fully fit) {fast path} else {slow path}
                  // tile_width is always exact. For the last block,
                  // it will be the exact number of elements left.
