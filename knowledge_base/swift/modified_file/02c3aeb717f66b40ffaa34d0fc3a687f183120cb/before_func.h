  static ValueTy &findOrAllocateNode_rec(NodeTy *P, KeyTy Key) {
    // Found the node we were looking for.
    if (P->Key == Key)
      return P->Payload;

    // Point to the edge we want to replace.
    typename ConcurrentMapNode<KeyTy, ValueTy>::EdgeTy *Edge = nullptr;

    // Select the edge to follow.
    if (P->Key > Key) {
      Edge = &P->Left;
    } else {
      Edge = &P->Right;
    }

    // Load the current edge value.
    ConcurrentMapNode<KeyTy, ValueTy> *CurrentVal =
      Edge->load(std::memory_order_acquire);

    // If the edge is populated the follow the edge.
    if (CurrentVal)
      return findOrAllocateNode_rec(CurrentVal, Key);

    // Allocate a new node.
    ConcurrentMapNode<KeyTy, ValueTy> *New =
        new ConcurrentMapNode<KeyTy, ValueTy>(Key);

    // Try to set a new node:
    if (std::atomic_compare_exchange_weak_explicit(Edge, &CurrentVal, New,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed)){
      // On success return the new node.
      return New->Payload;
    }

    // If failed, deallocate the node and look for a new place in the
    // tree. Some other thread may have created a new entry and we may
    // discover it, so start searching with the current node.
    delete New;
    return findOrAllocateNode_rec(P, Key);
  }
