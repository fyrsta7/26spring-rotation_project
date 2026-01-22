                wqitem->ndb_instance->db->getNdbError().message);
    DEBUG_ASSERT(false);
  }
  
  if(wqitem->base.verb == OPERATION_REPLACE) {
    DEBUG_PRINT(" [REPLACE] \"%.*s\"", wqitem->base.nkey, wqitem->key);
    ndb_op = op.updateTuple(tx);
  }
  else if(wqitem->base.verb == OPERATION_ADD) {
    DEBUG_PRINT(" [ADD]     \"%.*s\"", wqitem->base.nkey, wqitem->key);
    ndb_op = op.insertTuple(tx);
  }
  else if(wqitem->base.verb == OPERATION_CAS) {    
    if(server_cas) {
      /* NdbOperation.hpp says: "All data is copied out of the OperationOptions 
       structure (and any subtended structures) at operation definition time."      
       */
      DEBUG_PRINT(" [CAS UPDATE:%llu]     \"%.*s\"", cas_in, wqitem->base.nkey, wqitem->key);
      const Uint32 program_size = 25;
      Uint32 program[program_size];
      NdbInterpretedCode cas_code(plan->table, program, program_size);
