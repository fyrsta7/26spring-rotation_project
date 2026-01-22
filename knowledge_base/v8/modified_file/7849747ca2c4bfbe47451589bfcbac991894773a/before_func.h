class V8_EXPORT_PRIVATE TopLevelLiveRange final : public LiveRange {
 public:
  explicit TopLevelLiveRange(int vreg, MachineRepresentation rep);
  int spill_start_index() const { return spill_start_index_; }

  bool IsFixed() const { return vreg_ < 0; }

  bool is_phi() const { return IsPhiField::decode(bits_); }
  void set_is_phi(bool value) { bits_ = IsPhiField::update(bits_, value); }

  bool is_non_loop_phi() const { return IsNonLoopPhiField::decode(bits_); }
  void set_is_non_loop_phi(bool value) {
    bits_ = IsNonLoopPhiField::update(bits_, value);
  }

  bool has_slot_use() const { return HasSlotUseField::decode(bits_); }
  void set_has_slot_use(bool value) {
    bits_ = HasSlotUseField::update(bits_, value);
  }

  // Add a new interval or a new use position to this live range.
  void EnsureInterval(LifetimePosition start, LifetimePosition end, Zone* zone);
  void AddUseInterval(LifetimePosition start, LifetimePosition end, Zone* zone);
  void AddUsePosition(UsePosition* pos);

  // Shorten the most recently added interval by setting a new start.
  void ShortenTo(LifetimePosition start);

  // Detaches between start and end, and attributes the resulting range to
  // result.
  // The current range is pointed to as "splintered_from". No parent/child
  // relationship is established between this and result.
  void Splinter(LifetimePosition start, LifetimePosition end, Zone* zone);

  // Assuming other was splintered from this range, embeds other and its
  // children as part of the children sequence of this range.
  void Merge(TopLevelLiveRange* other, Zone* zone);

  // Spill range management.
  void SetSpillRange(SpillRange* spill_range);
  enum class SpillType { kNoSpillType, kSpillOperand, kSpillRange };
  void set_spill_type(SpillType value) {
    bits_ = SpillTypeField::update(bits_, value);
  }
  SpillType spill_type() const { return SpillTypeField::decode(bits_); }
  InstructionOperand* GetSpillOperand() const {
    DCHECK_EQ(SpillType::kSpillOperand, spill_type());
    return spill_operand_;
  }

  SpillRange* GetAllocatedSpillRange() const {
    DCHECK_NE(SpillType::kSpillOperand, spill_type());
    return spill_range_;
  }

  SpillRange* GetSpillRange() const {
    DCHECK_EQ(SpillType::kSpillRange, spill_type());
    return spill_range_;
  }
  bool HasNoSpillType() const {
    return spill_type() == SpillType::kNoSpillType;
  }
  bool HasSpillOperand() const {
    return spill_type() == SpillType::kSpillOperand;
  }
  bool HasSpillRange() const { return spill_type() == SpillType::kSpillRange; }

  AllocatedOperand GetSpillRangeOperand() const;

  void RecordSpillLocation(Zone* zone, int gap_index,
                           InstructionOperand* operand);
  void SetSpillOperand(InstructionOperand* operand);
  void SetSpillStartIndex(int start) {
    spill_start_index_ = Min(start, spill_start_index_);
  }

  void CommitSpillMoves(InstructionSequence* sequence,
                        const InstructionOperand& operand,
                        bool might_be_duplicated);

  // If all the children of this range are spilled in deferred blocks, and if
  // for any non-spilled child with a use position requiring a slot, that range
  // is contained in a deferred block, mark the range as
  // IsSpilledOnlyInDeferredBlocks, so that we avoid spilling at definition,
  // and instead let the LiveRangeConnector perform the spills within the
  // deferred blocks. If so, we insert here spills for non-spilled ranges
  // with slot use positions.
  void TreatAsSpilledInDeferredBlock(Zone* zone, int total_block_count) {
    spill_start_index_ = -1;
    spilled_in_deferred_blocks_ = true;
    spill_move_insertion_locations_ = nullptr;
    list_of_blocks_requiring_spill_operands_ =
        new (zone) BitVector(total_block_count, zone);
  }

  void CommitSpillInDeferredBlocks(RegisterAllocationData* data,
                                   const InstructionOperand& spill_operand,
                                   BitVector* necessary_spill_points);

  TopLevelLiveRange* splintered_from() const { return splintered_from_; }
  bool IsSplinter() const { return splintered_from_ != nullptr; }
  bool MayRequireSpillRange() const {
    DCHECK(!IsSplinter());
    return !HasSpillOperand() && spill_range_ == nullptr;
  }
  void UpdateSpillRangePostMerge(TopLevelLiveRange* merged);
  int vreg() const { return vreg_; }

#if DEBUG
  int debug_virt_reg() const;
#endif

  void Verify() const;
  void VerifyChildrenInOrder() const;

  int GetNextChildId() {
    return IsSplinter() ? splintered_from()->GetNextChildId()
                        : ++last_child_id_;
  }

  int GetChildCount() const { return last_child_id_ + 1; }

  bool IsSpilledOnlyInDeferredBlocks() const {
    return spilled_in_deferred_blocks_;
  }

  struct SpillMoveInsertionList;

  SpillMoveInsertionList* GetSpillMoveInsertionLocations() const {
    DCHECK(!IsSpilledOnlyInDeferredBlocks());
    return spill_move_insertion_locations_;
  }
  TopLevelLiveRange* splinter() const { return splinter_; }
  void SetSplinter(TopLevelLiveRange* splinter) {
    DCHECK_NULL(splinter_);
    DCHECK_NOT_NULL(splinter);

    splinter_ = splinter;
    splinter->relative_id_ = GetNextChildId();
    splinter->set_spill_type(spill_type());
    splinter->SetSplinteredFrom(this);
  }

  void MarkHasPreassignedSlot() { has_preassigned_slot_ = true; }
  bool has_preassigned_slot() const { return has_preassigned_slot_; }

  void AddBlockRequiringSpillOperand(RpoNumber block_id) {
    DCHECK(IsSpilledOnlyInDeferredBlocks());
    GetListOfBlocksRequiringSpillOperands()->Add(block_id.ToInt());
  }

  BitVector* GetListOfBlocksRequiringSpillOperands() const {
    DCHECK(IsSpilledOnlyInDeferredBlocks());
    return list_of_blocks_requiring_spill_operands_;
  }

 private:
  void SetSplinteredFrom(TopLevelLiveRange* splinter_parent);

  typedef BitField<bool, 1, 1> HasSlotUseField;
  typedef BitField<bool, 2, 1> IsPhiField;
  typedef BitField<bool, 3, 1> IsNonLoopPhiField;
  typedef BitField<SpillType, 4, 2> SpillTypeField;

  int vreg_;
  int last_child_id_;
  TopLevelLiveRange* splintered_from_;
  union {
    // Correct value determined by spill_type()
    InstructionOperand* spill_operand_;
    SpillRange* spill_range_;
  };

  union {
    SpillMoveInsertionList* spill_move_insertion_locations_;
    BitVector* list_of_blocks_requiring_spill_operands_;
  };

  // TODO(mtrofin): generalize spilling after definition, currently specialized
  // just for spill in a single deferred block.
  bool spilled_in_deferred_blocks_;
  int spill_start_index_;
  UsePosition* last_pos_;
  TopLevelLiveRange* splinter_;
  bool has_preassigned_slot_;

  DISALLOW_COPY_AND_ASSIGN(TopLevelLiveRange);
};
