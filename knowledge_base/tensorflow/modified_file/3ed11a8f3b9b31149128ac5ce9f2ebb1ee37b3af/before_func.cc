#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Determines whether an HLO instruction is replicated at index based on current
// knowledge in hlo_replication.
HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::DetermineHloInstructionIsReplicated(
    const HloInstruction* hlo, const ShapeIndex& index,
    bool cross_partition_spmd,
    const absl::flat_hash_map<const HloInstruction*, ShapeTree<HloReplication>>&
        hlo_replication,
    bool support_partial_replication) {
  const auto merge_operand_replication = [&hlo_replication](
                                             const HloInstruction* inst) {
    HloReplication replication = HloReplication::ReplicatedOnAllDevices();
    for (auto operand : inst->operands()) {
      auto operand_it = hlo_replication.find(operand);
      if (operand_it == hlo_replication.end()) {
        replication = replication.Merge(HloReplication::UniqueOnAllDevices());
      } else {
        replication = replication.Merge(operand_it->second.element({}));
      }
    }
    return replication;
  };

  if (hlo->opcode() == HloOpcode::kAllReduce ||
      hlo->opcode() == HloOpcode::kAllGather) {
    // All-reduce/all-gather returns same values across partitions/replicas as
    // long as its operands are replicated.
    HloReplication replication = merge_operand_replication(hlo);
    if (replication.IsReplicatedOnAllDevices()) {
      return replication;
    }
    if (!hlo->channel_id().has_value()) {
      // This is cross-replica-only.
      if (cross_partition_spmd) {
        return replication;
      }
      if (hlo->replica_groups().empty() || hlo->replica_groups().size() == 1) {
        return HloReplication::ReplicatedOnAllDevices();
      }
      if (support_partial_replication) {
        std::vector<absl::Span<const int64_t>> device_sets;
        for (const ReplicaGroup& replica_group : hlo->replica_groups()) {
          device_sets.push_back(replica_group.replica_ids());
        }
        return HloReplication::PartiallyReplicated(device_sets);
      } else {
        return HloReplication::UniqueOnAllDevices();
      }
    } else {
      bool global_id;
      if (hlo->opcode() == HloOpcode::kAllReduce) {
        global_id = Cast<HloAllReduceInstruction>(hlo)->use_global_device_ids();
      } else {
        global_id = Cast<HloAllGatherInstruction>(hlo)->use_global_device_ids();
      }
      if (global_id) {
        bool replicated_across_partitions = true;
        bool replicated_across_replicas = true;
        const int64_t num_partitions =
            hlo->GetModule()->config().num_partitions();
        for (const auto& group : hlo->replica_groups()) {
          absl::flat_hash_set<int64_t> visited_partitions;
          absl::flat_hash_set<int64_t> visited_replicas;
          for (int64_t id : group.replica_ids()) {
            int64_t rid = id / num_partitions;
            int64_t pid = id % num_partitions;
            visited_partitions.insert(pid);
            visited_replicas.insert(rid);
          }
          replicated_across_partitions &=
              visited_partitions.size() == num_partitions;
          replicated_across_replicas &=
              visited_replicas.size() ==
              hlo->GetModule()->config().replica_count();
        }
        if ((cross_partition_spmd && replicated_across_partitions) ||
            (!cross_partition_spmd && replicated_across_replicas)) {
          return HloReplication::ReplicatedOnAllDevices();
        } else {
          return HloReplication::UniqueOnAllDevices();
        }
      }
      if (cross_partition_spmd) {
        return HloReplication::ReplicatedOnAllDevices();
      }
      if (hlo->replica_groups().empty() || hlo->replica_groups().size() == 1) {
        return HloReplication::ReplicatedOnAllDevices();
      } else {
        return HloReplication::UniqueOnAllDevices();
      }
    }
  }
  if (hlo->HasSideEffectNoRecurse()) {
    return HloReplication::UniqueOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kReplicaId) {
    // ReplicaId returns the same value for all partitions in each replica.
    return cross_partition_spmd ? HloReplication::ReplicatedOnAllDevices()
                                : HloReplication::UniqueOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kPartitionId) {
    // PartitionId returns the same value for all replicas in each partition.
    return cross_partition_spmd ? HloReplication::UniqueOnAllDevices()
                                : HloReplication::ReplicatedOnAllDevices();
  }
  auto it = hlo_replication.find(hlo);
  if (hlo->opcode() == HloOpcode::kParameter) {
    // Parameters should have been processed.
    CHECK(it != hlo_replication.end());
    return it->second.element(index);
  }
  if (it != hlo_replication.end() &&
      it->second.element(index).IsUniqueOnAllDevices()) {
    // The HLO is already marked as non-replicated.
    return it->second.element(index);
  }

  if (hlo->opcode() == HloOpcode::kConstant) {
    return HloReplication::ReplicatedOnAllDevices();
  }

  if (hlo->opcode() == HloOpcode::kCustomCall &&
      (hlo->custom_call_target() == "X64SplitLow" ||
       hlo->custom_call_target() == "X64SplitHigh" ||
       hlo->custom_call_target() == "X64Combine")) {
    return merge_operand_replication(hlo);
  }

  // Pattern-match and process cases where the HLO is partially replicated.
  if (support_partial_replication) {
    // Below is a very specific pattern to match the SPMD pipeline case.
    if (hlo->opcode() == HloOpcode::kDynamicSlice) {
      const HloInstruction* ds_buffer = hlo->operand(0);
      if (hlo->dynamic_slice_sizes().size() == 1 &&
          hlo->dynamic_slice_sizes()[0] == 1 &&
          ds_buffer->opcode() == HloOpcode::kConstant &&
          ds_buffer->shape().rank() == 1 &&
          ds_buffer->shape().element_type() == PrimitiveType::S32 &&
          ((cross_partition_spmd &&
            hlo->operand(1)->opcode() == HloOpcode::kPartitionId) ||
           (!cross_partition_spmd &&
            hlo->operand(1)->opcode() == HloOpcode::kReplicaId))) {
        const HloModule* hlo_module = hlo->GetModule();
        int64_t num_devices = cross_partition_spmd
                                  ? hlo_module->config().num_partitions()
                                  : hlo_module->config().replica_count();
        absl::flat_hash_map<int64_t, std::vector<int64_t>> value_to_device_set;
        for (int64_t device_id = 0; device_id < num_devices; ++device_id) {
          std::optional<int64_t> value =
              ds_buffer->literal().GetIntegralAsS64({device_id});
          value_to_device_set[*value].push_back(device_id);
        }
        std::vector<absl::Span<const int64_t>> device_sets;
        for (const auto& value_and_device_set : value_to_device_set) {
          device_sets.push_back(
              absl::Span<const int64_t>(value_and_device_set.second));
        }
        return HloReplication::PartiallyReplicated(device_sets);
      }
    }
  }

  if (hlo->IsElementwise() ||                             //
      hlo->opcode() == HloOpcode::kConcatenate ||         //
      hlo->opcode() == HloOpcode::kConvolution ||         //
      hlo->opcode() == HloOpcode::kDot ||                 //
      hlo->opcode() == HloOpcode::kReduce ||              //
      hlo->opcode() == HloOpcode::kBroadcast ||           //
      hlo->opcode() == HloOpcode::kTranspose ||           //
      hlo->opcode() == HloOpcode::kReshape ||             //
      hlo->opcode() == HloOpcode::kBitcast ||             //
      hlo->opcode() == HloOpcode::kReverse ||             //
