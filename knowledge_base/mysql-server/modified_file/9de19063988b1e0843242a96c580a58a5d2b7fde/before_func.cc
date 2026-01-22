                      sequence_number, last_committed));

  if (first_event)
  {
    first_event= false;
  }
  else
  {
    if (unlikely(clock_leq(sequence_number, last_committed) &&
                 last_committed != SEQ_UNINIT))
    {
      /* inconsistent (buggy) timestamps */
      sql_print_error("Transaction is tagged with inconsistent logical "
                      "timestamps: "
                      "sequence_number (%lld) <= last_committed (%lld)",
                      sequence_number, last_committed);
      DBUG_RETURN(ER_MTS_CANT_PARALLEL);
    }
    if (unlikely(clock_leq(sequence_number, last_sequence_number) &&
                 sequence_number != SEQ_UNINIT))
    {
      /* inconsistent (buggy) timestamps */
      sql_print_error("Transaction's sequence number is inconsistent with that "
                      "of a preceding one: "
                      "sequence_number (%lld) <= previous sequence_number (%lld)",
                      sequence_number, last_sequence_number);
      DBUG_RETURN(ER_MTS_CANT_PARALLEL);
    }
    /*
      Being scheduled transaction sequence may have gaps, even in
      relay log. In such case a transaction that succeeds a gap will
      wait for all ealier that were scheduled to finish. It's marked
      as gap successor now.
    */
    compile_time_assert(SEQ_UNINIT == 0);
    if (unlikely(sequence_number > last_sequence_number + 1))
    {
      DBUG_PRINT("info", ("sequence_number gap found, "
                          "last_sequence_number %lld, sequence_number %lld",
                          last_sequence_number, sequence_number));
      DBUG_ASSERT(rli->replicate_same_server_id || true /* TODO: account autopositioning */);
      gap_successor= true;
    }
  }

  /*
    The new group flag is practically the same as the force flag
    when up to indicate syncronization with Workers.
  */
  is_new_group=
    (/* First event after a submode switch; */
     first_event ||
     /* Require a fresh group to be started; */
     // todo: turn `force_new_group' into sequence_number == SEQ_UNINIT condition
     force_new_group ||
     /* Rewritten event without commit point timestamp (todo: find use case) */
     sequence_number == SEQ_UNINIT ||
     /*
       undefined parent (e.g the very first trans from the master),
       or old master.
     */
     last_committed == SEQ_UNINIT ||
     /*
       When gap successor depends on a gap before it the scheduler has
       to serialize this transaction execution with previously
       scheduled ones. Below for simplicity it's assumed that such
       gap-dependency is always the case.
     */
     gap_successor ||
     /*
       previous group did not have sequence number assigned.
       It's execution must be finished until the current group
       can be assigned.
       Dependency of the current group on the previous
       can't be tracked. So let's wait till the former is over.
