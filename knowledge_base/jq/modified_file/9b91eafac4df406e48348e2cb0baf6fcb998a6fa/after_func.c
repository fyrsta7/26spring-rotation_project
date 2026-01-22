}

block gen_both(block a, block b) {
  block jump = gen_op_targetlater(JUMP);
  block fork = gen_op_target(FORK, jump);
  block c = BLOCK(fork, a, jump, b);
  inst_set_target(jump, c);
  return c;
}


block gen_collect(block expr) {
  block array_var = block_bind(gen_op_var_unbound(STOREV, "collect"),
                               gen_noop(), OP_HAS_VARIABLE);
