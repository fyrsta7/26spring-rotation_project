        }
    }
    close(fd);
    return true;
}

static ErrorOr<bool> parse_and_run(JS::Interpreter& interpreter, StringView source, StringView source_name)
{
    enum class ReturnEarly {
        No,
        Yes,
    };

    JS::ThrowCompletionOr<JS::Value> result { JS::js_undefined() };

    auto run_script_or_module = [&](auto& script_or_module) {
        if (s_dump_ast)
            script_or_module->parse_node().dump(0);

        if (JS::Bytecode::g_dump_bytecode || s_run_bytecode) {
            auto executable_result = JS::Bytecode::Generator::generate(script_or_module->parse_node());
            if (executable_result.is_error()) {
                result = g_vm->throw_completion<JS::InternalError>(executable_result.error().to_string());
                return ReturnEarly::No;
            }

            auto executable = executable_result.release_value();
            executable->name = source_name;
            if (s_opt_bytecode) {
                auto& passes = JS::Bytecode::Interpreter::optimization_pipeline(JS::Bytecode::Interpreter::OptimizationLevel::Optimize);
                passes.perform(*executable);
                dbgln("Optimisation passes took {}us", passes.elapsed());
            }

            if (JS::Bytecode::g_dump_bytecode)
                executable->dump();

            if (s_run_bytecode) {
                JS::Bytecode::Interpreter bytecode_interpreter(interpreter.realm());
                auto result_or_error = bytecode_interpreter.run_and_return_frame(*executable, nullptr);
                if (result_or_error.value.is_error())
                    result = result_or_error.value.release_error();
                else
                    result = result_or_error.frame->registers[0];
            } else {
                return ReturnEarly::Yes;
            }
        } else {
            result = interpreter.run(*script_or_module);
        }

        return ReturnEarly::No;
    };

    if (!s_as_module) {
        auto script_or_error = JS::Script::parse(source, interpreter.realm(), source_name);
        if (script_or_error.is_error()) {
            auto error = script_or_error.error()[0];
            auto hint = error.source_location_hint(source);
            if (!hint.is_empty())
                outln("{}", hint);
            outln("{}", error.to_string());
            result = interpreter.vm().throw_completion<JS::SyntaxError>(error.to_string());
        } else {
            auto return_early = run_script_or_module(script_or_error.value());
            if (return_early == ReturnEarly::Yes)
                return true;
        }
    } else {
        auto module_or_error = JS::SourceTextModule::parse(source, interpreter.realm(), source_name);
        if (module_or_error.is_error()) {
            auto error = module_or_error.error()[0];
            auto hint = error.source_location_hint(source);
            if (!hint.is_empty())
                outln("{}", hint);
            outln(error.to_string());
            result = interpreter.vm().throw_completion<JS::SyntaxError>(error.to_string());
        } else {
            auto return_early = run_script_or_module(module_or_error.value());
            if (return_early == ReturnEarly::Yes)
                return true;
        }
    }

    auto handle_exception = [&](JS::Value thrown_value) -> ErrorOr<void> {
        warnln("Uncaught exception: ");
        TRY(print(thrown_value, PrintTarget::StandardError));
        warnln();

        if (!thrown_value.is_object() || !is<JS::Error>(thrown_value.as_object()))
            return {};
        auto& traceback = static_cast<JS::Error const&>(thrown_value.as_object()).traceback();
        if (traceback.size() > 1) {
            unsigned repetitions = 0;
            for (size_t i = 0; i < traceback.size(); ++i) {
                auto& traceback_frame = traceback[i];
                if (i + 1 < traceback.size()) {
                    auto& next_traceback_frame = traceback[i + 1];
                    if (next_traceback_frame.function_name == traceback_frame.function_name) {
                        repetitions++;
                        continue;
                    }
                }
                if (repetitions > 4) {
                    // If more than 5 (1 + >4) consecutive function calls with the same name, print
                    // the name only once and show the number of repetitions instead. This prevents
                    // printing ridiculously large call stacks of recursive functions.
                    warnln(" -> {}", traceback_frame.function_name);
                    warnln(" {} more calls", repetitions);
                } else {
                    for (size_t j = 0; j < repetitions + 1; ++j)
                        warnln(" -> {}", traceback_frame.function_name);
                }
                repetitions = 0;
            }
        }
        return {};
    };

    if (!result.is_error())
        g_last_value = JS::make_handle(result.value());

    if (result.is_error()) {
        VERIFY(result.throw_completion().value().has_value());
        TRY(handle_exception(*result.release_error().value()));
        return false;
    }

    if (s_print_last_result) {
