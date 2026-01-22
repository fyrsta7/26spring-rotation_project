 *      Local<v8::FunctionTemplate> method_template =
 *        v8::FunctionTemplate::New(
 *          isolate, CustomEmbedderType::SlowMethod, v8::Local<v8::Value>(),
 *          v8::Local<v8::Signature>(), 1, v8::ConstructorBehavior::kAllow,
 *          v8::SideEffectType::kHasSideEffect, &c_func);
 *
 *      v8::Local<v8::ObjectTemplate> object_template =
 *        v8::ObjectTemplate::New(isolate);
