
    return ExecutionResult::Continue;
}

ALWAYS_INLINE ExecutionResult OpCode_Compare::execute(MatchInput const& input, MatchState& state) const
{
    bool inverse { false };
    bool temporary_inverse { false };
    bool reset_temp_inverse { false };
    struct DisjunctionState {
        bool active { false };
        bool is_conjunction { false };
        bool fail { false };
        size_t initial_position;
        size_t initial_code_unit_position;
        Optional<size_t> last_accepted_position {};
        Optional<size_t> last_accepted_code_unit_position {};
    };

    Vector<DisjunctionState, 4> disjunction_states;
    disjunction_states.empend();

    auto current_disjunction_state = [&]() -> DisjunctionState& { return disjunction_states.last(); };

    auto current_inversion_state = [&]() -> bool { return temporary_inverse ^ inverse; };

    size_t string_position = state.string_position;
    bool inverse_matched { false };
    bool had_zero_length_match { false };

    state.string_position_before_match = state.string_position;

    size_t offset { state.instruction_position + 3 };
    for (size_t i = 0; i < arguments_count(); ++i) {
        if (state.string_position > string_position)
            break;

        if (reset_temp_inverse) {
            reset_temp_inverse = false;
            temporary_inverse = false;
        } else {
            reset_temp_inverse = true;
        }

        auto compare_type = (CharacterCompareType)m_bytecode->at(offset++);

        if (compare_type == CharacterCompareType::Inverse) {
            inverse = !inverse;
            continue;

        } else if (compare_type == CharacterCompareType::TemporaryInverse) {
            // If "TemporaryInverse" is given, negate the current inversion state only for the next opcode.
            // it follows that this cannot be the last compare element.
            VERIFY(i != arguments_count() - 1);

            temporary_inverse = true;
            reset_temp_inverse = false;
            continue;

        } else if (compare_type == CharacterCompareType::Char) {
            u32 ch = m_bytecode->at(offset++);

            // We want to compare a string that is longer or equal in length to the available string
            if (input.view.length() <= state.string_position)
                return ExecutionResult::Failed_ExecuteLowPrioForks;

            compare_char(input, state, ch, current_inversion_state(), inverse_matched);

        } else if (compare_type == CharacterCompareType::AnyChar) {
            // We want to compare a string that is definitely longer than the available string
            if (input.view.length() <= state.string_position)
                return ExecutionResult::Failed_ExecuteLowPrioForks;

            auto input_view = input.view.substring_view(state.string_position, 1)[0];
            auto is_equivalent_to_newline = input_view == '\n'
                || (input.regex_options.has_flag_set(AllFlags::Internal_ECMA262DotSemantics)
                        ? (input_view == '\r' || input_view == LineSeparator || input_view == ParagraphSeparator)
                        : false);

            if (!is_equivalent_to_newline || (input.regex_options.has_flag_set(AllFlags::SingleLine) && input.regex_options.has_flag_set(AllFlags::Internal_ConsiderNewline))) {
                if (current_inversion_state())
                    inverse_matched = true;
                else
                    advance_string_position(state, input.view, input_view);
            }

        } else if (compare_type == CharacterCompareType::String) {
            VERIFY(!current_inversion_state());

            auto const& length = m_bytecode->at(offset++);

            // We want to compare a string that is definitely longer than the available string
            if (input.view.length() < state.string_position + length)
                return ExecutionResult::Failed_ExecuteLowPrioForks;

            Optional<DeprecatedString> str;
            Utf16Data utf16;
            Vector<u32> data;
            data.ensure_capacity(length);
            for (size_t i = offset; i < offset + length; ++i)
                data.unchecked_append(m_bytecode->at(i));

            auto view = input.view.construct_as_same(data, str, utf16);
            offset += length;
            if (compare_string(input, state, view, had_zero_length_match)) {
                if (current_inversion_state())
                    inverse_matched = true;
            }

        } else if (compare_type == CharacterCompareType::CharClass) {

            if (input.view.length() <= state.string_position_in_code_units)
                return ExecutionResult::Failed_ExecuteLowPrioForks;

            auto character_class = (CharClass)m_bytecode->at(offset++);
            auto ch = input.view[state.string_position_in_code_units];

            compare_character_class(input, state, character_class, ch, current_inversion_state(), inverse_matched);

        } else if (compare_type == CharacterCompareType::LookupTable) {
            if (input.view.length() <= state.string_position)
                return ExecutionResult::Failed_ExecuteLowPrioForks;

            auto count = m_bytecode->at(offset++);
            auto range_data = m_bytecode->template spans<4>().slice(offset, count);
            offset += count;

            auto ch = input.view[state.string_position_in_code_units];

            auto const* matching_range = binary_search(range_data, ch, nullptr, [insensitive = input.regex_options & AllFlags::Insensitive](auto needle, CharRange range) {
                auto upper_case_needle = needle;
                auto lower_case_needle = needle;
                if (insensitive) {
                    upper_case_needle = to_ascii_uppercase(needle);
                    lower_case_needle = to_ascii_lowercase(needle);
                }

                if (lower_case_needle >= range.from && lower_case_needle <= range.to)
                    return 0;
                if (upper_case_needle >= range.from && upper_case_needle <= range.to)
                    return 0;
                if (lower_case_needle > range.to || upper_case_needle > range.to)
                    return 1;
                return -1;
            });

            if (matching_range) {
                if (current_inversion_state())
                    inverse_matched = true;
                else
                    advance_string_position(state, input.view, ch);
            }

        } else if (compare_type == CharacterCompareType::CharRange) {
            if (input.view.length() <= state.string_position)
                return ExecutionResult::Failed_ExecuteLowPrioForks;

            auto value = (CharRange)m_bytecode->at(offset++);

            auto from = value.from;
            auto to = value.to;
            auto ch = input.view[state.string_position_in_code_units];

            compare_character_range(input, state, from, to, ch, current_inversion_state(), inverse_matched);

        } else if (compare_type == CharacterCompareType::Reference) {
            auto reference_number = (size_t)m_bytecode->at(offset++);
            auto& groups = state.capture_group_matches.at(input.match_index);
            if (groups.size() <= reference_number)
                return ExecutionResult::Failed_ExecuteLowPrioForks;

            auto str = groups.at(reference_number).view;

            // We want to compare a string that is definitely longer than the available string
            if (input.view.length() < state.string_position + str.length())
                return ExecutionResult::Failed_ExecuteLowPrioForks;

            if (compare_string(input, state, str, had_zero_length_match)) {
                if (current_inversion_state())
                    inverse_matched = true;
            }

        } else if (compare_type == CharacterCompareType::Property) {
            auto property = static_cast<Unicode::Property>(m_bytecode->at(offset++));
            compare_property(input, state, property, current_inversion_state(), inverse_matched);

        } else if (compare_type == CharacterCompareType::GeneralCategory) {
            auto general_category = static_cast<Unicode::GeneralCategory>(m_bytecode->at(offset++));
            compare_general_category(input, state, general_category, current_inversion_state(), inverse_matched);

        } else if (compare_type == CharacterCompareType::Script) {
            auto script = static_cast<Unicode::Script>(m_bytecode->at(offset++));
            compare_script(input, state, script, current_inversion_state(), inverse_matched);

        } else if (compare_type == CharacterCompareType::ScriptExtension) {
            auto script = static_cast<Unicode::Script>(m_bytecode->at(offset++));
            compare_script_extension(input, state, script, current_inversion_state(), inverse_matched);

        } else if (compare_type == CharacterCompareType::And) {
            disjunction_states.append({
                .active = true,
                .is_conjunction = false,
                .fail = false,
                .initial_position = state.string_position,
                .initial_code_unit_position = state.string_position_in_code_units,
            });
            continue;

        } else if (compare_type == CharacterCompareType::Or) {
            disjunction_states.append({
                .active = true,
                .is_conjunction = true,
                .fail = true,
                .initial_position = state.string_position,
                .initial_code_unit_position = state.string_position_in_code_units,
            });
            continue;

        } else if (compare_type == CharacterCompareType::EndAndOr) {
            auto disjunction_state = disjunction_states.take_last();
            if (!disjunction_state.fail) {
                state.string_position = disjunction_state.last_accepted_position.value_or(disjunction_state.initial_position);
                state.string_position_in_code_units = disjunction_state.last_accepted_code_unit_position.value_or(disjunction_state.initial_code_unit_position);
            }

        } else {
            warnln("Undefined comparison: {}", (int)compare_type);
            VERIFY_NOT_REACHED();
            break;
        }

        auto& new_disjunction_state = current_disjunction_state();
        if (current_inversion_state() && (!inverse || new_disjunction_state.active) && !inverse_matched) {
            advance_string_position(state, input.view);
            inverse_matched = true;
        }

        if (new_disjunction_state.active) {
            auto failed = (!had_zero_length_match && string_position == state.string_position) || state.string_position > input.view.length();

            if (!failed) {
                new_disjunction_state.last_accepted_position = state.string_position;
                new_disjunction_state.last_accepted_code_unit_position = state.string_position_in_code_units;
            }

            if (new_disjunction_state.is_conjunction)
                new_disjunction_state.fail = failed && new_disjunction_state.fail;
            else
                new_disjunction_state.fail = failed || new_disjunction_state.fail;

            state.string_position = new_disjunction_state.initial_position;
            state.string_position_in_code_units = new_disjunction_state.initial_code_unit_position;
        }
    }

    auto& new_disjunction_state = current_disjunction_state();
    if (new_disjunction_state.active) {
        if (!new_disjunction_state.fail) {
            state.string_position = new_disjunction_state.last_accepted_position.value_or(new_disjunction_state.initial_position);
            state.string_position_in_code_units = new_disjunction_state.last_accepted_code_unit_position.value_or(new_disjunction_state.initial_code_unit_position);
        }
    }

    if (current_inversion_state() && !inverse_matched)
        advance_string_position(state, input.view);

    if ((!had_zero_length_match && string_position == state.string_position) || state.string_position > input.view.length())
