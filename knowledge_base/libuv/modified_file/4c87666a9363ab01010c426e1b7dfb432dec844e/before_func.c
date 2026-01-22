
int uv_async_init(uv_loop_t* loop, uv_async_t* handle, uv_async_cb async_cb) {
  if (uv__async_init(loop))
    return uv__set_sys_error(loop, errno);

  uv__handle_init(loop, (uv_handle_t*)handle, UV_ASYNC);
  loop->counters.async_init++;

  handle->async_cb = async_cb;
  handle->pending = 0;

  ngx_queue_insert_tail(&loop->async_handles, &handle->queue);
  uv__handle_start(handle);

