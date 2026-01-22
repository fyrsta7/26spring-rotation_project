
  if (auto fusible = IsProducerConsumerFusible(*producer, *consumer);
      !fusible) {
    return fusible;
  }

  if (CreatesHeavyComputation(*producer, *consumer)) {
    return "the fusion would create a heavy computation";
  }

  return InstructionFusion::ShouldFuse(consumer, operand_index);
}

