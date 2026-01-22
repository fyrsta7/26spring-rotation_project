}

NonnullGCPtr<ECMAScriptFunctionObject> ECMAScriptFunctionObject::create(Realm& realm, DeprecatedFlyString name, Object& prototype, ByteString source_text, Statement const& ecmascript_code, Vector<FunctionParameter> parameters, i32 m_function_length, Vector<DeprecatedFlyString> local_variables_names, Environment* parent_environment, PrivateEnvironment* private_environment, FunctionKind kind, bool is_strict, bool might_need_arguments_object, bool contains_direct_call_to_eval, bool is_arrow_function, Variant<PropertyKey, PrivateName, Empty> class_field_initializer_name)
{
    return realm.heap().allocate<ECMAScriptFunctionObject>(realm, move(name), move(source_text), ecmascript_code, move(parameters), m_function_length, move(local_variables_names), parent_environment, private_environment, prototype, kind, is_strict, might_need_arguments_object, contains_direct_call_to_eval, is_arrow_function, move(class_field_initializer_name));
}

ECMAScriptFunctionObject::ECMAScriptFunctionObject(DeprecatedFlyString name, ByteString source_text, Statement const& ecmascript_code, Vector<FunctionParameter> formal_parameters, i32 function_length, Vector<DeprecatedFlyString> local_variables_names, Environment* parent_environment, PrivateEnvironment* private_environment, Object& prototype, FunctionKind kind, bool strict, bool might_need_arguments_object, bool contains_direct_call_to_eval, bool is_arrow_function, Variant<PropertyKey, PrivateName, Empty> class_field_initializer_name)
    : FunctionObject(prototype)
    , m_name(move(name))
    , m_function_length(function_length)
    , m_local_variables_names(move(local_variables_names))
    , m_environment(parent_environment)
    , m_private_environment(private_environment)
    , m_formal_parameters(move(formal_parameters))
    , m_ecmascript_code(ecmascript_code)
    , m_realm(&prototype.shape().realm())
    , m_source_text(move(source_text))
    , m_class_field_initializer_name(move(class_field_initializer_name))
    , m_strict(strict)
    , m_might_need_arguments_object(might_need_arguments_object)
    , m_contains_direct_call_to_eval(contains_direct_call_to_eval)
    , m_is_arrow_function(is_arrow_function)
    , m_kind(kind)
{
    // NOTE: This logic is from OrdinaryFunctionCreate, https://tc39.es/ecma262/#sec-ordinaryfunctioncreate

    // 9. If thisMode is lexical-this, set F.[[ThisMode]] to lexical.
    if (m_is_arrow_function)
        m_this_mode = ThisMode::Lexical;
    // 10. Else if Strict is true, set F.[[ThisMode]] to strict.
    else if (m_strict)
        m_this_mode = ThisMode::Strict;
    else
        // 11. Else, set F.[[ThisMode]] to global.
        m_this_mode = ThisMode::Global;

    // 15. Set F.[[ScriptOrModule]] to GetActiveScriptOrModule().
    m_script_or_module = vm().get_active_script_or_module();

    // 15.1.3 Static Semantics: IsSimpleParameterList, https://tc39.es/ecma262/#sec-static-semantics-issimpleparameterlist
    m_has_simple_parameter_list = all_of(m_formal_parameters, [&](auto& parameter) {
        if (parameter.is_rest)
            return false;
        if (parameter.default_value)
            return false;
        if (!parameter.binding.template has<NonnullRefPtr<Identifier const>>())
            return false;
        return true;
    });

    // NOTE: The following steps are from FunctionDeclarationInstantiation that could be executed once
    //       and then reused in all subsequent function instantiations.

    // 2. Let code be func.[[ECMAScriptCode]].
    ScopeNode const* scope_body = nullptr;
    if (is<ScopeNode>(*m_ecmascript_code))
        scope_body = static_cast<ScopeNode const*>(m_ecmascript_code.ptr());

    // 3. Let strict be func.[[Strict]].

    // 4. Let formals be func.[[FormalParameters]].
    auto const& formals = m_formal_parameters;

    // 5. Let parameterNames be the BoundNames of formals.
    // 6. If parameterNames has any duplicate entries, let hasDuplicates be true. Otherwise, let hasDuplicates be false.

    size_t parameters_in_environment = 0;

    // NOTE: This loop performs step 5, 6, and 8.
    for (auto const& parameter : formals) {
        if (parameter.default_value)
            m_has_parameter_expressions = true;

        parameter.binding.visit(
            [&](Identifier const& identifier) {
                if (m_parameter_names.set(identifier.string(), identifier.is_local() ? ParameterIsLocal::Yes : ParameterIsLocal::No) != AK::HashSetResult::InsertedNewEntry)
                    m_has_duplicates = true;
                else if (!identifier.is_local())
                    ++parameters_in_environment;
            },
            [&](NonnullRefPtr<BindingPattern const> const& pattern) {
                if (pattern->contains_expression())
                    m_has_parameter_expressions = true;

                // NOTE: Nothing in the callback throws an exception.
                MUST(pattern->for_each_bound_identifier([&](auto& identifier) {
                    if (m_parameter_names.set(identifier.string(), identifier.is_local() ? ParameterIsLocal::Yes : ParameterIsLocal::No) != AK::HashSetResult::InsertedNewEntry)
                        m_has_duplicates = true;
                    else if (!identifier.is_local())
                        ++parameters_in_environment;
                }));
            });
    }

    // 15. Let argumentsObjectNeeded be true.
    m_arguments_object_needed = m_might_need_arguments_object;

    // 16. If func.[[ThisMode]] is lexical, then
    if (this_mode() == ThisMode::Lexical) {
        // a. NOTE: Arrow functions never have an arguments object.
        // b. Set argumentsObjectNeeded to false.
        m_arguments_object_needed = false;
    }
    // 17. Else if parameterNames contains "arguments", then
    else if (m_parameter_names.contains(vm().names.arguments.as_string())) {
        // a. Set argumentsObjectNeeded to false.
        m_arguments_object_needed = false;
    }

    HashTable<DeprecatedFlyString> function_names;

    // 18. Else if hasParameterExpressions is false, then
    //     a. If functionNames contains "arguments" or lexicalNames contains "arguments", then
    //         i. Set argumentsObjectNeeded to false.
    // NOTE: The block below is a combination of step 14 and step 18.
    if (scope_body) {
        // NOTE: Nothing in the callback throws an exception.
        MUST(scope_body->for_each_var_function_declaration_in_reverse_order([&](FunctionDeclaration const& function) {
            if (function_names.set(function.name()) == AK::HashSetResult::InsertedNewEntry)
                m_functions_to_initialize.append(function);
        }));

        auto const& arguments_name = vm().names.arguments.as_string();

        if (!m_has_parameter_expressions && function_names.contains(arguments_name))
            m_arguments_object_needed = false;

        if (!m_has_parameter_expressions && m_arguments_object_needed) {
            // NOTE: Nothing in the callback throws an exception.
            MUST(scope_body->for_each_lexically_declared_identifier([&](auto const& identifier) {
                if (identifier.string() == arguments_name)
                    m_arguments_object_needed = false;
            }));
        }
    } else {
        m_arguments_object_needed = false;
    }

    size_t* environment_size = nullptr;

    size_t parameter_environment_bindings_count = 0;
    // 19. If strict is true or hasParameterExpressions is false, then
    if (m_strict || !m_has_parameter_expressions) {
        // a. NOTE: Only a single Environment Record is needed for the parameters, since calls to eval in strict mode code cannot create new bindings which are visible outside of the eval.
        // b. Let env be the LexicalEnvironment of calleeContext
        // NOTE: Here we are only interested in the size of the environment.
        environment_size = &m_function_environment_bindings_count;
    }
    // 20. Else,
    else {
        // a. NOTE: A separate Environment Record is needed to ensure that bindings created by direct eval calls in the formal parameter list are outside the environment where parameters are declared.
        // b. Let calleeEnv be the LexicalEnvironment of calleeContext.
        // c. Let env be NewDeclarativeEnvironment(calleeEnv).
        environment_size = &parameter_environment_bindings_count;
    }

    *environment_size += parameters_in_environment;

    HashMap<DeprecatedFlyString, ParameterIsLocal> parameter_bindings;

    // 22. If argumentsObjectNeeded is true, then
    if (m_arguments_object_needed) {
        // f. Let parameterBindings be the list-concatenation of parameterNames and « "arguments" ».
        parameter_bindings = m_parameter_names;
        parameter_bindings.set(vm().names.arguments.as_string(), ParameterIsLocal::No);

        (*environment_size)++;
    } else {
        parameter_bindings = m_parameter_names;
        // a. Let parameterBindings be parameterNames.
    }

    HashMap<DeprecatedFlyString, ParameterIsLocal> instantiated_var_names;

    size_t* var_environment_size = nullptr;

    // 27. If hasParameterExpressions is false, then
    if (!m_has_parameter_expressions) {
        // b. Let instantiatedVarNames be a copy of the List parameterBindings.
        instantiated_var_names = parameter_bindings;

        if (scope_body) {
            // c. For each element n of varNames, do
            MUST(scope_body->for_each_var_declared_identifier([&](auto const& id) {
                // i. If instantiatedVarNames does not contain n, then
                if (instantiated_var_names.set(id.string(), id.is_local() ? ParameterIsLocal::Yes : ParameterIsLocal::No) == AK::HashSetResult::InsertedNewEntry) {
                    // 1. Append n to instantiatedVarNames.
                    // Following steps will be executed in function_declaration_instantiation:
                    // 2. Perform ! env.CreateMutableBinding(n, false).
                    // 3. Perform ! env.InitializeBinding(n, undefined).
                    m_var_names_to_initialize_binding.append({
                        .identifier = id,
                        .parameter_binding = parameter_bindings.contains(id.string()),
                        .function_name = function_names.contains(id.string()),
                    });

                    if (!id.is_local())
                        (*environment_size)++;
                }
            }));
        }

        // d. Let varEnv be env
        var_environment_size = environment_size;
    } else {
        // a. NOTE: A separate Environment Record is needed to ensure that closures created by expressions in the formal parameter list do not have visibility of declarations in the function body.

        // b. Let varEnv be NewDeclarativeEnvironment(env).
        // NOTE: Here we are only interested in the size of the environment.
        var_environment_size = &m_var_environment_bindings_count;

        // 28. Else,
        // NOTE: Steps a, b, c and d are executed in function_declaration_instantiation.
        // e. For each element n of varNames, do
        if (scope_body) {
            MUST(scope_body->for_each_var_declared_identifier([&](auto const& id) {
                // 1. Append n to instantiatedVarNames.
                // Following steps will be executed in function_declaration_instantiation:
                // 2. Perform ! env.CreateMutableBinding(n, false).
                // 3. Perform ! env.InitializeBinding(n, undefined).
                if (instantiated_var_names.set(id.string(), id.is_local() ? ParameterIsLocal::Yes : ParameterIsLocal::No) == AK::HashSetResult::InsertedNewEntry) {
                    m_var_names_to_initialize_binding.append({
                        .identifier = id,
                        .parameter_binding = parameter_bindings.contains(id.string()),
                        .function_name = function_names.contains(id.string()),
                    });

                    if (!id.is_local())
                        (*var_environment_size)++;
                }
            }));
        }
    }

    // 29. NOTE: Annex B.3.2.1 adds additional steps at this point.
    // B.3.2.1 Changes to FunctionDeclarationInstantiation, https://tc39.es/ecma262/#sec-web-compat-functiondeclarationinstantiation
    if (!m_strict && scope_body) {
        MUST(scope_body->for_each_function_hoistable_with_annexB_extension([&](FunctionDeclaration& function_declaration) {
            auto function_name = function_declaration.name();
            if (parameter_bindings.contains(function_name))
                return;

            if (!instantiated_var_names.contains(function_name) && function_name != vm().names.arguments.as_string()) {
                m_function_names_to_initialize_binding.append(function_name);
                instantiated_var_names.set(function_name, ParameterIsLocal::No);
                (*var_environment_size)++;
            }

            function_declaration.set_should_do_additional_annexB_steps();
        }));
    }

    size_t* lex_environment_size = nullptr;

    // 30. If strict is false, then
    if (!m_strict) {
        bool can_elide_declarative_environment = !m_contains_direct_call_to_eval && (!scope_body || !scope_body->has_lexical_declarations());
        if (can_elide_declarative_environment) {
            lex_environment_size = var_environment_size;
        } else {
            // a. Let lexEnv be NewDeclarativeEnvironment(varEnv).
            lex_environment_size = &m_lex_environment_bindings_count;
        }
    } else {
        // a. let lexEnv be varEnv.
        // NOTE: Here we are only interested in the size of the environment.
        lex_environment_size = var_environment_size;
    }

