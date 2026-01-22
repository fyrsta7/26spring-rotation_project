	void Callback() override {
		auto &statement = Get<Statement>();
		auto env = statement.Env();
		Napi::HandleScope scope(env);

		auto cb = callback.Value();
		if (!statement.statement->success) {
			cb.MakeCallback(statement.Value(), {Utils::CreateError(env, statement.statement->error)});
			return;
		}
		cb.MakeCallback(statement.Value(), {env.Null(), statement.Value()});
	}
