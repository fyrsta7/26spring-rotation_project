	void DoCallback() override {
		auto &statement = Get<Statement>();
		Napi::Env env = statement.Env();
		Napi::HandleScope scope(env);

		if (!statement.statement) {
			deferred.Reject(Utils::CreateError(env, "statement was finalized"));
		} else if (statement.statement->HasError()) {
			deferred.Reject(Utils::CreateError(env, statement.statement->GetError()));
		} else if (result->HasError()) {
			deferred.Reject(Utils::CreateError(env, result->GetError()));
		} else {
			auto db = statement.connection_ref->database_ref->Value();
			auto query_result = QueryResult::constructor.New({db});
			auto unwrapped = Napi::ObjectWrap<QueryResult>::Unwrap(query_result);
			unwrapped->result = move(result);
			deferred.Resolve(query_result);
		}
	}
