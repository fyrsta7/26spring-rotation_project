  void init(SILBasicBlock *NewBB, unsigned bitcnt, bool optimistic) {
    BB = NewBB;
    LocationNum = bitcnt;
    // For reachable basic blocks, the initial state of ForwardSetOut should be
    // all 1's. Otherwise the dataflow solution could be too conservative.
    //
    // Consider this case, the forwardable value by var a = 10 before the loop
    // will not be forwarded if the ForwardSetOut is set to 0 initially.
    //
    //   var a = 10
    //   for _ in 0...1024 {}
    //   use(a);
    //
    // However, by doing so, we can only do the data forwarding after the
    // data flow stabilizes.
    //
    // We set the initial state of unreachable block to 0, as we do not have
    // a value for the location.
    //
    // This is a bit conservative as we could be missing forwarding
    // opportunities. i.e. a joint block with 1 predecessor being an
    // unreachable block.
    //
    // we rely on other passes to clean up unreachable block.
    ForwardSetIn.resize(LocationNum, false);
    ForwardSetOut.resize(LocationNum, optimistic);

    // If we are running an optimsitic data flow, set forward max to true
    // initially.
    ForwardSetMax.resize(LocationNum, optimistic);

    BBGenSet.resize(LocationNum, false);
    BBKillSet.resize(LocationNum, false);
  }
