  if (ret < 0) {
    return nullptr;
  }

  if (is_s3website) {
    if (s->init_state.url_bucket.empty()) {
      return new RGWHandler_REST_Service_S3Website(auth_registry);
    }
    if (rgw::sal::Object::empty(s->object.get())) {
      return new RGWHandler_REST_Bucket_S3Website(auth_registry);
    }
    return new RGWHandler_REST_Obj_S3Website(auth_registry);
  }

  if (s->init_state.url_bucket.empty()) {
    // no bucket
    if (s->op == OP_POST) {
      // POST will be one of: IAM, STS or topic service
      const auto max_size = s->cct->_conf->rgw_max_put_param_size;
      int ret;
      bufferlist data;
      std::tie(ret, data) = rgw_rest_read_all_input(s, max_size, false);
      if (ret < 0) {
        return nullptr;
      }
      parse_post_action(data.to_str(), s);
