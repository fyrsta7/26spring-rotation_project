      if (first_error == GRPC_ERROR_NONE) {
        first_error = error;
      } else {
        GRPC_ERROR_UNREF(error);
      }
    }
    user_data +=
        GPR_ROUND_UP_TO_ALIGNMENT_SIZE(filters[i]->sizeof_channel_data);
    call_size += GPR_ROUND_UP_TO_ALIGNMENT_SIZE(filters[i]->sizeof_call_data);
  }

  GPR_ASSERT(user_data > (char*)stack);
  GPR_ASSERT((uintptr_t)(user_data - (char*)stack) ==
             grpc_channel_stack_size(filters, filter_count));

  stack->call_stack_size = call_size;
  return first_error;
}

void grpc_channel_stack_destroy(grpc_channel_stack* stack) {
  grpc_channel_element* channel_elems = CHANNEL_ELEMS_FROM_STACK(stack);
  size_t count = stack->count;
  size_t i;

  /* destroy per-filter data */
  for (i = 0; i < count; i++) {
    channel_elems[i].filter->destroy_channel_elem(&channel_elems[i]);
  }
}

grpc_error* grpc_call_stack_init(grpc_channel_stack* channel_stack,
                                 int initial_refs, grpc_iomgr_cb_func destroy,
                                 void* destroy_arg,
                                 const grpc_call_element_args* elem_args) {
  grpc_channel_element* channel_elems = CHANNEL_ELEMS_FROM_STACK(channel_stack);
  size_t count = channel_stack->count;
  grpc_call_element* call_elems;
  char* user_data;
