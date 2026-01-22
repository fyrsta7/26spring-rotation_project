}


void RGWRESTOp::send_response()
{
  if (!flusher.did_start()) {
    set_req_state_err(s, get_ret());
    dump_errno(s);
    end_header(s, this);
  }
  flusher.flush();
}

int RGWRESTOp::verify_permission(optional_yield)
{
  return check_caps(s->user->get_info().caps);
}

RGWOp* RGWHandler_REST::get_op(void)
{
  RGWOp *op;
  switch (s->op) {
   case OP_GET:
     op = op_get();
     break;
   case OP_PUT:
     op = op_put();
     break;
   case OP_DELETE:
     op = op_delete();
     break;
   case OP_HEAD:
     op = op_head();
