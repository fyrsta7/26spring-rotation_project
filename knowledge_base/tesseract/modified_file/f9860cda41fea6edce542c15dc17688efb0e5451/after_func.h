
class IntParam : public Param {
 public:
  IntParam(int32_t value, const char* name, const char* comment, bool init,
           ParamsVectors* vec)
      : Param(name, comment, init) {
    value_ = value;
    default_ = value;
    params_vec_ = &(vec->int_params);
    vec->int_params.push_back(this);
