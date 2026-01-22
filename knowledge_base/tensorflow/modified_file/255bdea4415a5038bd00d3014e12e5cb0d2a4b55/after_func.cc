
  if (auto fusible = IsProducerConsumerFusible(*producer, *consumer);
      !fusible) {
    return fusible;
  }

  if (CreatesHeavyComputation(*producer, *consumer)) {
    return "the fusion would create a heavy computation";
  }

  return InstructionFusion::ShouldFuse(consumer, operand_index);
}

FusionDecision GpuPriorityFusion::ShouldFuse(HloInstruction* consumer,
                                             int64_t operand_index) {
  if (auto fusible = ShouldFuseInexpensiveChecks(consumer, operand_index);
      !fusible) {
    return fusible;
  }

  auto producer = consumer->operand(operand_index);

  // The following checks are potentially expensive.
  if (auto fusible = FusionFitsInBudget(*consumer, *producer, device_info_,
                                        /*is_consumer_producer_fusion=*/true);
      !fusible) {
    return fusible;
  }

