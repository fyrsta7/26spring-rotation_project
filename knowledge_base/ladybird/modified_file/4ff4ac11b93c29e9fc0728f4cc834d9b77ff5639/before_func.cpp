    // 1. Set F.[[HomeObject]] to homeObject.
    m_home_object = &home_object;

    // 2. Return unused.
}

// 10.2.11 FunctionDeclarationInstantiation ( func, argumentsList ), https://tc39.es/ecma262/#sec-functiondeclarationinstantiation
ThrowCompletionOr<void> ECMAScriptFunctionObject::function_declaration_instantiation()
{
    auto& vm = this->vm();
    auto& realm = *vm.current_realm();

    // 1. Let calleeContext be the running execution context.
    auto& callee_context = vm.running_execution_context();

    // 2. Let code be func.[[ECMAScriptCode]].
    ScopeNode const* scope_body = nullptr;
    if (is<ScopeNode>(*m_ecmascript_code))
        scope_body = static_cast<ScopeNode const*>(m_ecmascript_code.ptr());

    // NOTE: Following steps were executed in ECMAScriptFunctionObject constructor.
    //       3. Let strict be func.[[Strict]].
    //       4. Let formals be func.[[FormalParameters]].
    //       5. Let parameterNames be the BoundNames of formals.
    //       6. If parameterNames has any duplicate entries, let hasDuplicates be true. Otherwise, let hasDuplicates be false.

    // 7. Let simpleParameterList be IsSimpleParameterList of formals.
    bool const simple_parameter_list = has_simple_parameter_list();

    // NOTE: Following steps were executed in ECMAScriptFunctionObject constructor.
    //       8. Let hasParameterExpressions be ContainsExpression of formals.
    //       9. Let varNames be the VarDeclaredNames of code.
    //       10. Let varDeclarations be the VarScopedDeclarations of code.
    //       11. Let lexicalNames be the LexicallyDeclaredNames of code.
    //       12. Let functionNames be a new empty List.
    //       13. Let functionsToInitialize be a new empty List.
    //       14. For each element d of varDeclarations, in reverse List order, do
    //       15. Let argumentsObjectNeeded be true.
    //       16. If func.[[ThisMode]] is lexical, then
    //       17. Else if parameterNames contains "arguments", then
    //       18. Else if hasParameterExpressions is false, then

    GCPtr<Environment> environment;

    // 19. If strict is true or hasParameterExpressions is false, then
    if (m_strict || !m_has_parameter_expressions) {
        // a. NOTE: Only a single Environment Record is needed for the parameters, since calls to eval in strict mode code cannot create new bindings which are visible outside of the eval.
        // b. Let env be the LexicalEnvironment of calleeContext.
        environment = callee_context.lexical_environment;
    }
    // 20. Else,
    else {
        // a. NOTE: A separate Environment Record is needed to ensure that bindings created by direct eval calls in the formal parameter list are outside the environment where parameters are declared.

        // b. Let calleeEnv be the LexicalEnvironment of calleeContext.
        auto callee_env = callee_context.lexical_environment;

        // c. Let env be NewDeclarativeEnvironment(calleeEnv).
        environment = new_declarative_environment(*callee_env);

        // d. Assert: The VariableEnvironment of calleeContext is calleeEnv.
        VERIFY(callee_context.variable_environment == callee_context.lexical_environment);

        // e. Set the LexicalEnvironment of calleeContext to env.
        callee_context.lexical_environment = environment;
    }

    // 21. For each String paramName of parameterNames, do
    for (auto const& parameter_name : m_parameter_names) {
        // a. Let alreadyDeclared be ! env.HasBinding(paramName).
        auto already_declared = MUST(environment->has_binding(parameter_name));

        // b. NOTE: Early errors ensure that duplicate parameter names can only occur in non-strict functions that do not have parameter default values or rest parameters.

        // c. If alreadyDeclared is false, then
        if (!already_declared) {
            // i. Perform ! env.CreateMutableBinding(paramName, false).
            MUST(environment->create_mutable_binding(vm, parameter_name, false));

            // ii. If hasDuplicates is true, then
            if (m_has_duplicates) {
                // 1. Perform ! env.InitializeBinding(paramName, undefined).
                MUST(environment->initialize_binding(vm, parameter_name, js_undefined(), Environment::InitializeBindingHint::Normal));
            }
        }
    }

    // 22. If argumentsObjectNeeded is true, then
    if (m_arguments_object_needed) {
        Object* arguments_object;

        // a. If strict is true or simpleParameterList is false, then
        if (m_strict || !simple_parameter_list) {
            // i. Let ao be CreateUnmappedArgumentsObject(argumentsList).
            arguments_object = create_unmapped_arguments_object(vm, vm.running_execution_context().arguments);
        }
        // b. Else,
        else {
            // i. NOTE: A mapped argument object is only provided for non-strict functions that don't have a rest parameter, any parameter default value initializers, or any destructured parameters.

            // ii. Let ao be CreateMappedArgumentsObject(func, formals, argumentsList, env).
            arguments_object = create_mapped_arguments_object(vm, *this, formal_parameters(), vm.running_execution_context().arguments, *environment);
        }

        // c. If strict is true, then
        if (m_strict) {
            // i. Perform ! env.CreateImmutableBinding("arguments", false).
            MUST(environment->create_immutable_binding(vm, vm.names.arguments.as_string(), false));

            // ii. NOTE: In strict mode code early errors prevent attempting to assign to this binding, so its mutability is not observable.
        }
        // b. Else,
        else {
            // i. Perform ! env.CreateMutableBinding("arguments", false).
            MUST(environment->create_mutable_binding(vm, vm.names.arguments.as_string(), false));
        }

        // c. Perform ! env.InitializeBinding("arguments", ao).
        MUST(environment->initialize_binding(vm, vm.names.arguments.as_string(), arguments_object, Environment::InitializeBindingHint::Normal));

        // f. Let parameterBindings be the list-concatenation of parameterNames and « "arguments" ».
    }
    // 23. Else,
    else {
        // a. Let parameterBindings be parameterNames.
    }

    // NOTE: We now treat parameterBindings as parameterNames.

    // 24. Let iteratorRecord be CreateListIteratorRecord(argumentsList).
    // 25. If hasDuplicates is true, then
    //     a. Perform ? IteratorBindingInitialization of formals with arguments iteratorRecord and undefined.
    // 26. Else,
    //     a. Perform ? IteratorBindingInitialization of formals with arguments iteratorRecord and env.
    // NOTE: The spec makes an iterator here to do IteratorBindingInitialization but we just do it manually
    auto& execution_context_arguments = vm.running_execution_context().arguments;

    size_t default_parameter_index = 0;
    for (size_t i = 0; i < m_formal_parameters.size(); ++i) {
        auto& parameter = m_formal_parameters[i];
        if (parameter.default_value)
            ++default_parameter_index;

        TRY(parameter.binding.visit(
            [&](auto const& param) -> ThrowCompletionOr<void> {
                Value argument_value;
                if (parameter.is_rest) {
                    auto array = MUST(Array::create(realm, 0));
                    for (size_t rest_index = i; rest_index < execution_context_arguments.size(); ++rest_index)
                        array->indexed_properties().append(execution_context_arguments[rest_index]);
                    argument_value = array;
                } else if (i < execution_context_arguments.size() && !execution_context_arguments[i].is_undefined()) {
                    argument_value = execution_context_arguments[i];
                } else if (parameter.default_value) {
                    auto value_and_frame = vm.bytecode_interpreter().run_and_return_frame(realm, *m_default_parameter_bytecode_executables[default_parameter_index - 1], nullptr);
                    if (value_and_frame.value.is_error())
                        return value_and_frame.value.release_error();
                    // Resulting value is in the accumulator.
                    argument_value = value_and_frame.frame->registers.at(0);
                } else {
                    argument_value = js_undefined();
                }

                Environment* used_environment = m_has_duplicates ? nullptr : environment;

                if constexpr (IsSame<NonnullRefPtr<Identifier const> const&, decltype(param)>) {
                    if (param->is_local()) {
                        callee_context.local_variables[param->local_variable_index()] = argument_value;
                        return {};
                    }
                    Reference reference = TRY(vm.resolve_binding(param->string(), used_environment));
                    // Here the difference from hasDuplicates is important
                    if (m_has_duplicates)
                        return reference.put_value(vm, argument_value);
                    return reference.initialize_referenced_binding(vm, argument_value);
                }
                if constexpr (IsSame<NonnullRefPtr<BindingPattern const> const&, decltype(param)>) {
                    // Here the difference from hasDuplicates is important
                    return vm.binding_initialization(param, argument_value, used_environment);
                }
            }));
    }

    GCPtr<Environment> var_environment;

    // 27. If hasParameterExpressions is false, then
    if (!m_has_parameter_expressions) {
        // a. NOTE: Only a single Environment Record is needed for the parameters and top-level vars.

        // b. Let instantiatedVarNames be a copy of the List parameterBindings.
        // NOTE: Done in implementation of step 27.c.i.1 below

        if (scope_body) {
            // NOTE: Due to the use of MUST with `create_mutable_binding` and `initialize_binding` below,
            //       an exception should not result from `for_each_var_declared_name`.

            // c. For each element n of varNames, do
            for (auto const& variable_to_initialize : m_var_names_to_initialize_binding) {
                auto const& id = variable_to_initialize.identifier;
                // NOTE: Following steps were executed in ECMAScriptFunctionObject constructor.
                //       i. If instantiatedVarNames does not contain n, then
                //       1. Append n to instantiatedVarNames.
                if (id.is_local()) {
                    callee_context.local_variables[id.local_variable_index()] = js_undefined();
                } else {
                    // 2. Perform ! env.CreateMutableBinding(n, false).
                    // 3. Perform ! env.InitializeBinding(n, undefined).
                    MUST(environment->create_mutable_binding(vm, id.string(), false));
                    MUST(environment->initialize_binding(vm, id.string(), js_undefined(), Environment::InitializeBindingHint::Normal));
                }
            }
        }

        // d.Let varEnv be env
        var_environment = environment;
    }
    // 28. Else,
    else {
        // a. NOTE: A separate Environment Record is needed to ensure that closures created by expressions in the formal parameter list do not have visibility of declarations in the function body.

        // b. Let varEnv be NewDeclarativeEnvironment(env).
        var_environment = new_declarative_environment(*environment);

        // c. Set the VariableEnvironment of calleeContext to varEnv.
        callee_context.variable_environment = var_environment;

        // d. Let instantiatedVarNames be a new empty List.
        // NOTE: Already done above.

        if (scope_body) {
            // NOTE: Due to the use of MUST with `create_mutable_binding`, `get_binding_value` and `initialize_binding` below,
            //       an exception should not result from `for_each_var_declared_name`.

            // e. For each element n of varNames, do
            for (auto const& variable_to_initialize : m_var_names_to_initialize_binding) {
                auto const& id = variable_to_initialize.identifier;

                // NOTE: Following steps were executed in ECMAScriptFunctionObject constructor.
                //       i. If instantiatedVarNames does not contain n, then
                //       1. Append n to instantiatedVarNames.

                // 2. Perform ! varEnv.CreateMutableBinding(n, false).
                MUST(var_environment->create_mutable_binding(vm, id.string(), false));

                Value initial_value;

                // 3. If parameterBindings does not contain n, or if functionNames contains n, then
                if (!variable_to_initialize.parameter_binding || variable_to_initialize.function_name) {
                    // a. Let initialValue be undefined.
                    initial_value = js_undefined();
                }
                // 4. Else,
                else {
                    // a. Let initialValue be ! env.GetBindingValue(n, false).
                    if (id.is_local()) {
                        initial_value = callee_context.local_variables[id.local_variable_index()];
                    } else {
                        initial_value = MUST(environment->get_binding_value(vm, id.string(), false));
                    }
                }

                // 5. Perform ! varEnv.InitializeBinding(n, initialValue).
                if (id.is_local()) {
                    // NOTE: Local variables are supported only in bytecode interpreter
                    callee_context.local_variables[id.local_variable_index()] = initial_value;
                } else {
                    MUST(var_environment->initialize_binding(vm, id.string(), initial_value, Environment::InitializeBindingHint::Normal));
                }

                // 6. NOTE: A var with the same name as a formal parameter initially has the same value as the corresponding initialized parameter.
            }
        }
    }

    // 29. NOTE: Annex B.3.2.1 adds additional steps at this point.
    // B.3.2.1 Changes to FunctionDeclarationInstantiation, https://tc39.es/ecma262/#sec-web-compat-functiondeclarationinstantiation
    if (!m_strict && scope_body) {
        // NOTE: Due to the use of MUST with `create_mutable_binding` and `initialize_binding` below,
        //       an exception should not result from `for_each_function_hoistable_with_annexB_extension`.
        for (auto const& function_name : m_function_names_to_initialize_binding) {
            MUST(var_environment->create_mutable_binding(vm, function_name, false));
            MUST(var_environment->initialize_binding(vm, function_name, js_undefined(), Environment::InitializeBindingHint::Normal));
        }
    }

    GCPtr<Environment> lex_environment;

    // 30. If strict is false, then
    if (!m_strict) {
        // Optimization: We avoid creating empty top-level declarative environments in non-strict mode, if both of these conditions are true:
        //               1. there is no direct call to eval() within this function
        //               2. there are no lexical declarations that would go into the environment
        bool can_elide_declarative_environment = !m_contains_direct_call_to_eval && (!scope_body || !scope_body->has_lexical_declarations());
        if (can_elide_declarative_environment) {
            lex_environment = var_environment;
        } else {
            // a. Let lexEnv be NewDeclarativeEnvironment(varEnv).
            // b. NOTE: Non-strict functions use a separate Environment Record for top-level lexical declarations so that a direct eval
            //          can determine whether any var scoped declarations introduced by the eval code conflict with pre-existing top-level
            //          lexically scoped declarations. This is not needed for strict functions because a strict direct eval always places
            //          all declarations into a new Environment Record.
            lex_environment = new_declarative_environment(*var_environment);
        }
    }
    // 31. Else,
    else {
        // a. let lexEnv be varEnv.
        lex_environment = var_environment;
    }

    // 32. Set the LexicalEnvironment of calleeContext to lexEnv.
    callee_context.lexical_environment = lex_environment;

    if (!scope_body)
        return {};

    // 33. Let lexDeclarations be the LexicallyScopedDeclarations of code.
    // 34. For each element d of lexDeclarations, do
    // NOTE: Due to the use of MUST in the callback, an exception should not result from `for_each_lexically_scoped_declaration`.
    MUST(scope_body->for_each_lexically_scoped_declaration([&](Declaration const& declaration) {
        // NOTE: Due to the use of MUST with `create_immutable_binding` and `create_mutable_binding` below,
        //       an exception should not result from `for_each_bound_name`.

        // a. NOTE: A lexically declared name cannot be the same as a function/generator declaration, formal parameter, or a var name. Lexically declared names are only instantiated here but not initialized.

        // b. For each element dn of the BoundNames of d, do
        MUST(declaration.for_each_bound_identifier([&](auto const& id) {
            if (id.is_local()) {
                // NOTE: Local variables are supported only in bytecode interpreter
                return;
            }

            // i. If IsConstantDeclaration of d is true, then
            if (declaration.is_constant_declaration()) {
                // 1. Perform ! lexEnv.CreateImmutableBinding(dn, true).
                MUST(lex_environment->create_immutable_binding(vm, id.string(), true));
            }
            // ii. Else,
            else {
                // 1. Perform ! lexEnv.CreateMutableBinding(dn, false).
                MUST(lex_environment->create_mutable_binding(vm, id.string(), false));
            }
        }));
    }));

    // 35. Let privateEnv be the PrivateEnvironment of calleeContext.
    auto private_environment = callee_context.private_environment;

    // 36. For each Parse Node f of functionsToInitialize, do
    for (auto& declaration : m_functions_to_initialize) {
        // a. Let fn be the sole element of the BoundNames of f.
        // b. Let fo be InstantiateFunctionObject of f with arguments lexEnv and privateEnv.
        auto function = ECMAScriptFunctionObject::create(realm, declaration.name(), declaration.source_text(), declaration.body(), declaration.parameters(), declaration.function_length(), declaration.local_variables_names(), lex_environment, private_environment, declaration.kind(), declaration.is_strict_mode(), declaration.might_need_arguments_object(), declaration.contains_direct_call_to_eval());

        // c. Perform ! varEnv.SetMutableBinding(fn, fo, false).
        if (declaration.name_identifier()->is_local()) {
            callee_context.local_variables[declaration.name_identifier()->local_variable_index()] = function;
        } else {
            MUST(var_environment->set_mutable_binding(vm, declaration.name(), function, false));
        }
    }

    if (is<DeclarativeEnvironment>(*lex_environment))
