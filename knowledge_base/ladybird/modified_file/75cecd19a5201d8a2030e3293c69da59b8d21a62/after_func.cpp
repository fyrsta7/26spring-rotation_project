// NOTE: Refrain from using the URL classes setters inside this algorithm. Rather, set the values directly. This bypasses the setters' built-in
//       validation, which is strictly unnecessary since we set m_valid=true at the end anyways. Furthermore, this algorithm may be used in the
//       future for validation of URLs, which would then lead to infinite recursion.
//       The same goes for base_url, because e.g. the port() getter does not always return m_port, and we are interested in the underlying member
//       variables' values here, not what the URL class presents to its users.
URL URLParser::basic_parse(StringView raw_input, Optional<URL> const& base_url, Optional<URL> url, Optional<State> state_override)
{
    dbgln_if(URL_PARSER_DEBUG, "URLParser::parse: Parsing '{}'", raw_input);
    if (raw_input.is_empty())
        return base_url.has_value() ? *base_url : URL {};

    size_t start_index = 0;
    size_t end_index = raw_input.length();

    // 1. If url is not given:
    if (!url.has_value()) {
        // 1. Set url to a new URL.
        url = URL();

        // 2. If input contains any leading or trailing C0 control or space, invalid-URL-unit validation error.
        // 3. Remove any leading and trailing C0 control or space from input.
        //
        // FIXME: We aren't checking exactly for 'trailing C0 control or space' here.

        bool has_validation_error = false;
        for (size_t i = 0; i < raw_input.length(); ++i) {
            i8 ch = raw_input[i];
            if (0 <= ch && ch <= 0x20) {
                ++start_index;
                has_validation_error = true;
            } else {
                break;
            }
        }
        for (ssize_t i = raw_input.length() - 1; i >= 0; --i) {
            i8 ch = raw_input[i];
            if (0 <= ch && ch <= 0x20) {
                --end_index;
                has_validation_error = true;
            } else {
                break;
            }
        }
        if (has_validation_error)
            report_validation_error();
    }
    if (start_index >= end_index)
        return {};

    ByteString processed_input = raw_input.substring_view(start_index, end_index - start_index);

    // 2. If input contains any ASCII tab or newline, invalid-URL-unit validation error.
    // 3. Remove all ASCII tab or newline from input.
    if (processed_input.contains("\t"sv) || processed_input.contains("\n"sv)) {
        report_validation_error();
        processed_input = processed_input.replace("\t"sv, ""sv, ReplaceMode::All).replace("\n"sv, ""sv, ReplaceMode::All);
    }

    // 4. Let state be state override if given, or scheme start state otherwise.
    State state = state_override.value_or(State::SchemeStart);

    // FIXME: 5. Set encoding to the result of getting an output encoding from encoding.

    // 6. Let buffer be the empty string.
    StringBuilder buffer;

    // 7. Let atSignSeen, insideBrackets, and passwordTokenSeen be false.
    bool at_sign_seen = false;
    bool inside_brackets = false;
    bool password_token_seen = false;

    Utf8View input(processed_input);

    // 8. Let pointer be a pointer for input.
    Utf8CodePointIterator iterator = input.begin();

    auto get_remaining = [&input, &iterator] {
        return input.substring_view(iterator - input.begin() + iterator.underlying_code_point_length_in_bytes()).as_string();
    };

    auto remaining_starts_with_two_ascii_hex_digits = [&]() {
        return is_ascii_hex_digit(iterator.peek(1).value_or(end_of_file)) && is_ascii_hex_digit(iterator.peek(2).value_or(end_of_file));
    };

    // 9. Keep running the following state machine by switching on state. If after a run pointer points to the EOF code point, go to the next step. Otherwise, increase pointer by 1 and continue with the state machine.
    // NOTE: "continue" should only be used to prevent incrementing the iterator, as this is done at the end of the loop.
    //       ++iterator : "increase pointer by 1"
    //       continue   : "decrease pointer by 1"
    for (;;) {
        u32 code_point = end_of_file;
        if (!iterator.done())
            code_point = *iterator;

        if constexpr (URL_PARSER_DEBUG) {
            if (code_point == end_of_file)
                dbgln("URLParser::basic_parse: {} state with EOF.", state_name(state));
            else if (is_ascii_printable(code_point))
                dbgln("URLParser::basic_parse: {} state with code point U+{:04X} ({:c}).", state_name(state), code_point, code_point);
            else
                dbgln("URLParser::basic_parse: {} state with code point U+{:04X}.", state_name(state), code_point);
        }

        switch (state) {
        // -> scheme start state, https://url.spec.whatwg.org/#scheme-start-state
        case State::SchemeStart:
            // 1. If c is an ASCII alpha, append c, lowercased, to buffer, and set state to scheme state.
            if (is_ascii_alpha(code_point)) {
                buffer.append_as_lowercase(code_point);
                state = State::Scheme;
            }
            // 2. Otherwise, if state override is not given, set state to no scheme state and decrease pointer by 1.
            else if (!state_override.has_value()) {
                state = State::NoScheme;
                continue;
            }
            // 3. Otherwise, return failure.
            else {
                return {};
            }
            break;
        // -> scheme state, https://url.spec.whatwg.org/#scheme-state
        case State::Scheme:
            // 1. If c is an ASCII alphanumeric, U+002B (+), U+002D (-), or U+002E (.), append c, lowercased, to buffer.
            if (is_ascii_alphanumeric(code_point) || code_point == '+' || code_point == '-' || code_point == '.') {
                buffer.append_as_lowercase(code_point);
            }
            // 2. Otherwise, if c is U+003A (:), then:
            else if (code_point == ':') {
                // 1. If state override is given, then:
                if (state_override.has_value()) {
                    // 1. If url’s scheme is a special scheme and buffer is not a special scheme, then return.
                    if (URL::is_special_scheme(url->scheme()) && !URL::is_special_scheme(buffer.string_view()))
                        return *url;

                    // 2. If url’s scheme is not a special scheme and buffer is a special scheme, then return.
                    if (!URL::is_special_scheme(url->scheme()) && URL::is_special_scheme(buffer.string_view()))
                        return *url;

                    // 3. If url includes credentials or has a non-null port, and buffer is "file", then return.
                    if ((url->includes_credentials() || url->port().has_value()) && buffer.string_view() == "file"sv)
                        return *url;

                    // 4. If url’s scheme is "file" and its host is an empty host, then return.
                    if (url->scheme() == "file"sv && url->host() == String {})
                        return *url;
                }

                // 2. Set url’s scheme to buffer.
                url->m_scheme = buffer.to_string_without_validation();

                // 3. If state override is given, then:
                if (state_override.has_value()) {
                    // 1. If url’s port is url’s scheme’s default port, then set url’s port to null.
                    if (url->port() == URL::default_port_for_scheme(url->scheme()))
                        url->m_port = {};

                    // 2. Return.
                    return *url;
                }

                // 4. Set buffer to the empty string.
                buffer.clear();

                // 5. If url’s scheme is "file", then:
                if (url->scheme() == "file") {
                    // 1. If remaining does not start with "//", special-scheme-missing-following-solidus validation error.
                    if (!get_remaining().starts_with("//"sv)) {
                        report_validation_error();
                    }
                    // 2. Set state to file state.
                    state = State::File;
                }
                // 6. Otherwise, if url is special, base is non-null, and base’s scheme is url’s scheme:
                else if (url->is_special() && base_url.has_value() && base_url->scheme() == url->m_scheme) {
                    // 1. Assert: base is is special (and therefore does not have an opaque path).
                    VERIFY(base_url->is_special());

                    // 2. Set state to special relative or authority state.
                    state = State::SpecialRelativeOrAuthority;
                }
                // 7. Otherwise, if url is special, set state to special authority slashes state.
                else if (url->is_special()) {
                    state = State::SpecialAuthoritySlashes;
                }
                // 8. Otherwise, if remaining starts with an U+002F (/), set state to path or authority state and increase pointer by 1.
                else if (get_remaining().starts_with("/"sv)) {
                    state = State::PathOrAuthority;
                    ++iterator;
                }
                // 9. Otherwise, set url’s path to the empty string and set state to opaque path state.
                else {
                    url->m_cannot_be_a_base_url = true;
                    url->append_slash();
                    state = State::CannotBeABaseUrlPath;
                }
            }
            // 3. Otherwise, if state override is not given, set buffer to the empty string, state to no scheme state, and start over (from the first code point in input).
            else if (!state_override.has_value()) {
                buffer.clear();
                state = State::NoScheme;
                iterator = input.begin();
                continue;
            }
            // 4. Otherwise, return failure.
            else {
                return {};
            }
            break;
        // -> no scheme state, https://url.spec.whatwg.org/#no-scheme-state
        case State::NoScheme:
            // 1. If base is null, or base has an opaque path and c is not U+0023 (#), missing-scheme-non-relative-URL validation error, return failure.
            if (!base_url.has_value() || (base_url->m_cannot_be_a_base_url && code_point != '#')) {
                report_validation_error();
                return {};
            }
            // 2. Otherwise, if base has an opaque path and c is U+0023 (#), set url’s scheme to base’s scheme, url’s path to base’s path, url’s query
            //    to base’s query,url’s fragment to the empty string, and set state to fragment state.
            else if (base_url->m_cannot_be_a_base_url && code_point == '#') {
                url->m_scheme = base_url->m_scheme;
                url->m_paths = base_url->m_paths;
                url->m_query = base_url->m_query;
                url->m_fragment = String {};
                url->m_cannot_be_a_base_url = true;
                state = State::Fragment;
            }
            // 3. Otherwise, if base’s scheme is not "file", set state to relative state and decrease pointer by 1.
            else if (base_url->m_scheme != "file") {
                state = State::Relative;
                continue;
            }
            // 4. Otherwise, set state to file state and decrease pointer by 1.
            else {
                state = State::File;
                continue;
            }
            break;
        // -> special relative or authority state, https://url.spec.whatwg.org/#special-relative-or-authority-state
        case State::SpecialRelativeOrAuthority:
            // 1. If c is U+002F (/) and remaining starts with U+002F (/), then set state to special authority ignore slashes state and increase pointer by 1.
            if (code_point == '/' && get_remaining().starts_with("/"sv)) {
                state = State::SpecialAuthorityIgnoreSlashes;
                ++iterator;
            }
            // 2. Otherwise, special-scheme-missing-following-solidus validation error, set state to relative state and decrease pointer by 1.
            else {
                report_validation_error();
                state = State::Relative;
                continue;
            }
            break;
        // -> path or authority state, https://url.spec.whatwg.org/#path-or-authority-state
        case State::PathOrAuthority:
            // 1. If c is U+002F (/), then set state to authority state.
            if (code_point == '/') {
                state = State::Authority;
            }
            // 2. Otherwise, set state to path state, and decrease pointer by 1.
            else {
                state = State::Path;
                continue;
            }
            break;
        // -> relative state, https://url.spec.whatwg.org/#relative-state
        case State::Relative:
            // 1. Assert: base’s scheme is not "file".
            VERIFY(base_url->scheme() != "file");

            // 2. Set url’s scheme to base’s scheme.
            url->m_scheme = base_url->m_scheme;

            // 3. If c is U+002F (/), then set state to relative slash state.
            if (code_point == '/') {
                state = State::RelativeSlash;
            }
            // 4. Otherwise, if url is special and c is U+005C (\), invalid-reverse-solidus validation error, set state to relative slash state.
            else if (url->is_special() && code_point == '\\') {
                report_validation_error();
                state = State::RelativeSlash;
            }
            // 5. Otherwise:
            else {
                // 1. Set url’s username to base’s username, url’s password to base’s password, url’s host to base’s host, url’s port to base’s port, url’s path to a clone of base’s path, and url’s query to base’s query.
                url->m_username = base_url->m_username;
                url->m_password = base_url->m_password;
                url->m_host = base_url->m_host;
                url->m_port = base_url->m_port;
                url->m_paths = base_url->m_paths;
                url->m_query = base_url->m_query;

                // 2. If c is U+003F (?), then set url’s query to the empty string, and state to query state.
                if (code_point == '?') {
                    url->m_query = String {};
                    state = State::Query;
                }
                // 3. Otherwise, if c is U+0023 (#), set url’s fragment to the empty string and state to fragment state.
                else if (code_point == '#') {
                    url->m_fragment = String {};
                    state = State::Fragment;
                }
                // 4. Otherwise, if c is not the EOF code point:
                else if (code_point != end_of_file) {
                    // 1. Set url’s query to null.
                    url->m_query = {};

                    // 2. Shorten url’s path.
                    shorten_urls_path(*url);

                    // 3. Set state to path state and decrease pointer by 1.
                    state = State::Path;
                    continue;
                }
            }
            break;
        // -> relative slash state, https://url.spec.whatwg.org/#relative-slash-state
        case State::RelativeSlash:
            // 1. If url is special and c is U+002F (/) or U+005C (\), then:
            if (url->is_special() && (code_point == '/' || code_point == '\\')) {
                // 1. If c is U+005C (\), invalid-reverse-solidus validation error.
                if (code_point == '\\')
                    report_validation_error();

                // 2. Set state to special authority ignore slashes state.
                state = State::SpecialAuthorityIgnoreSlashes;
            }
            // 2. Otherwise, if c is U+002F (/), then set state to authority state.
            else if (code_point == '/') {
                state = State::Authority;
            }
            // 3. Otherwise, set url’s username to base’s username, url’s password to base’s password, url’s host to base’s host, url’s port to base’s port, state to path state, and then, decrease pointer by 1.
            else {
                url->m_username = base_url->m_username;
                url->m_password = base_url->m_password;
                url->m_host = base_url->m_host;
                url->m_port = base_url->m_port;
                state = State::Path;
                continue;
            }
            break;
        // -> special authority slashes state, https://url.spec.whatwg.org/#special-authority-slashes-state
        case State::SpecialAuthoritySlashes:
            // 1. If c is U+002F (/) and remaining starts with U+002F (/), then set state to special authority ignore slashes state and increase pointer by 1.
            if (code_point == '/' && get_remaining().starts_with("/"sv)) {
                state = State::SpecialAuthorityIgnoreSlashes;
                ++iterator;
            }
            // 2. Otherwise, special-scheme-missing-following-solidus validation error, set state to special authority ignore slashes state and decrease pointer by 1.
            else {
                report_validation_error();
                state = State::SpecialAuthorityIgnoreSlashes;
                continue;
            }
            break;
        // -> special authority ignore slashes state, https://url.spec.whatwg.org/#special-authority-ignore-slashes-state
        case State::SpecialAuthorityIgnoreSlashes:
            // 1. If c is neither U+002F (/) nor U+005C (\), then set state to authority state and decrease pointer by 1.
            if (code_point != '/' && code_point != '\\') {
                state = State::Authority;
                continue;
            }
            // 2. Otherwise, special-scheme-missing-following-solidus validation error.
            else {
                report_validation_error();
            }
            break;
        // -> authority state, https://url.spec.whatwg.org/#authority-state
        case State::Authority:
            // 1. If c is U+0040 (@), then:
            if (code_point == '@') {
                // 1. Invalid-credentials validation error.
                report_validation_error();

                // 2. If atSignSeen is true, then prepend "%40" to buffer.
                if (at_sign_seen) {
                    auto content = buffer.to_byte_string();
                    buffer.clear();
                    buffer.append("%40"sv);
                    buffer.append(content);
                }

                // 3. Set atSignSeen to true.
                at_sign_seen = true;

                StringBuilder username_builder;
                StringBuilder password_builder;

                // 4. For each codePoint in buffer:
                for (auto c : Utf8View(buffer.string_view())) {
                    // 1. If codePoint is U+003A (:) and passwordTokenSeen is false, then set passwordTokenSeen to true and continue.
                    if (c == ':' && !password_token_seen) {
                        password_token_seen = true;
                        continue;
                    }

                    // 2. Let encodedCodePoints be the result of running UTF-8 percent-encode codePoint using the userinfo percent-encode set.
                    // NOTE: This is done inside of step 3 and 4 implementation

                    // 3. If passwordTokenSeen is true, then append encodedCodePoints to url’s password.
                    if (password_token_seen) {
                        if (password_builder.is_empty())
                            password_builder.append(url->m_password);

                        URL::append_percent_encoded_if_necessary(password_builder, c, URL::PercentEncodeSet::Userinfo);
                    }
                    // 4. Otherwise, append encodedCodePoints to url’s username.
                    else {
                        if (username_builder.is_empty())
                            username_builder.append(url->m_username);

                        URL::append_percent_encoded_if_necessary(username_builder, c, URL::PercentEncodeSet::Userinfo);
                    }
                }

                if (username_builder.string_view().length() > url->m_username.bytes().size())
                    url->m_username = username_builder.to_string().release_value_but_fixme_should_propagate_errors();
                if (password_builder.string_view().length() > url->m_password.bytes().size())
                    url->m_password = password_builder.to_string().release_value_but_fixme_should_propagate_errors();

                // 5. Set buffer to the empty string.
                buffer.clear();

            }
            // 2. Otherwise, if one of the following is true:
            //    * c is the EOF code point, U+002F (/), U+003F (?), or U+0023 (#)
            //    * url is special and c is U+005C (\)
            else if ((code_point == end_of_file || code_point == '/' || code_point == '?' || code_point == '#')
                || (url->is_special() && code_point == '\\')) {
                // then:

                // 1. If atSignSeen is true and buffer is the empty string, invalid-credentials validation error, return failure.
                if (at_sign_seen && buffer.is_empty()) {
                    report_validation_error();
                    return {};
                }

                // 2. Decrease pointer by buffer’s code point length + 1, set buffer to the empty string, and set state to host state.
                iterator = input.iterator_at_byte_offset(iterator - input.begin() - buffer.length() - 1);
                buffer.clear();
                state = State::Host;
            }
            // 3. Otherwise, append c to buffer.
            else {
                buffer.append_code_point(code_point);
            }
            break;
        // -> host state, https://url.spec.whatwg.org/#host-state
        // -> hostname state, https://url.spec.whatwg.org/#hostname-state
        case State::Host:
        case State::Hostname:
            // 1. If state override is given and url’s scheme is "file", then decrease pointer by 1 and set state to file host state.
            if (state_override.has_value() && url->scheme() == "file") {
                state = State::FileHost;
                continue;
            }

            // 2. Otherwise, if c is U+003A (:) and insideBrackets is false, then:
            if (code_point == ':' && !inside_brackets) {
                // 1. If buffer is the empty string, host-missing validation error, return failure.
                if (buffer.is_empty()) {
                    report_validation_error();
                    return {};
                }

                // 2. If state override is given and state override is hostname state, then return.
                if (state_override.has_value() && *state_override == State::Hostname)
                    return *url;

                // 3. Let host be the result of host parsing buffer with url is not special.
                auto host = parse_host(buffer.string_view(), !url->is_special());

                // 4. If host is failure, then return failure.
                if (!host.has_value())
                    return {};

                // 5. Set url’s host to host, buffer to the empty string, and state to port state.
                url->m_host = host.release_value();
                buffer.clear();
                state = State::Port;
            }
            // 3. Otherwise, if one of the following is true:
            //    * c is the EOF code point, U+002F (/), U+003F (?), or U+0023 (#)
            //    * url is special and c is U+005C (\)
            else if ((code_point == end_of_file || code_point == '/' || code_point == '?' || code_point == '#')
                || (url->is_special() && code_point == '\\')) {
                // then decrease pointer by 1, and then:
                // NOTE: pointer decrement is done by the continue below

                // 1. If url is special and buffer is the empty string, host-missing validation error, return failure.
                if (url->is_special() && buffer.is_empty()) {
                    report_validation_error();
                    return {};
                }

                // 2. Otherwise, if state override is given, buffer is the empty string, and either url includes credentials or url’s port is non-null, return.
                if (state_override.has_value() && buffer.is_empty() && (url->includes_credentials() || url->port().has_value()))
                    return *url;

                // 3. Let host be the result of host parsing buffer with url is not special.
                auto host = parse_host(buffer.string_view(), !url->is_special());

                // 4. If host is failure, then return failure.
                if (!host.has_value())
                    return {};

                // 5. Set url’s host to host, buffer to the empty string, and state to path start state.
                url->m_host = host.value();
                buffer.clear();
                state = State::Port;

                // 6. If state override is given, then return.
                if (state_override.has_value())
                    return *url;

                continue;

            }
            // 4. Otherwise:
            else {
                // 1. If c is U+005B ([), then set insideBrackets to true.
                if (code_point == '[') {
                    inside_brackets = true;
                }
                // 2. If c is U+005D (]), then set insideBrackets to false.
                else if (code_point == ']') {
                    inside_brackets = false;
                }

                // 3. Append c to buffer.
                buffer.append_code_point(code_point);
            }
            break;
        // -> port state, https://url.spec.whatwg.org/#port-state
        case State::Port:
            // 1. If c is an ASCII digit, append c to buffer.
            if (is_ascii_digit(code_point)) {
                buffer.append_code_point(code_point);
            }

            // 2. Otherwise, if one of the following is true:
            //    * c is the EOF code point, U+002F (/), U+003F (?), or U+0023 (#)
            //    * url is special and c is U+005C (\)
            //    * state override is given
            else if ((code_point == end_of_file || code_point == '/' || code_point == '?' || code_point == '#')
                || (url->is_special() && code_point == '\\')
                || state_override.has_value()) {
                // then:

                // 1. If buffer is not the empty string, then:
                if (!buffer.is_empty()) {
                    // 1. Let port be the mathematical integer value that is represented by buffer in radix-10 using ASCII digits for digits with values 0 through 9.
                    auto port = buffer.string_view().to_number<u16>();

                    // 2. If port is greater than 2^16 − 1, port-out-of-range validation error, return failure.
                    // NOTE: This is done by to_number.
                    if (!port.has_value()) {
                        report_validation_error();
                        return {};
                    }

                    // 3. Set url’s port to null, if port is url’s scheme’s default port; otherwise to port.
                    if (port.value() == URL::default_port_for_scheme(url->scheme()))
                        url->m_port = {};
                    else
                        url->m_port = port.value();

                    // 4. Set buffer to the empty string.
                    buffer.clear();
                }

                // 2. If state override is given, then return.
                if (state_override.has_value())
                    return *url;

                // 3. Set state to path start state and decrease pointer by 1.
                state = State::PathStart;
                continue;
            }
            // 3. Otherwise, port-invalid validation error, return failure.
            else {
                report_validation_error();
                return {};
            }
            break;
        // -> file state, https://url.spec.whatwg.org/#file-state
        case State::File:
            // 1. Set url’s scheme to "file".
            url->m_scheme = String::from_utf8("file"sv).release_value_but_fixme_should_propagate_errors();

            // 2. Set url’s host to the empty string.
            url->m_host = String {};

            // 3. If c is U+002F (/) or U+005C (\), then:
            if (code_point == '/' || code_point == '\\') {
                // 1. If c is U+005C (\), invalid-reverse-solidus validation error.
                if (code_point == '\\')
                    report_validation_error();

                // 2. Set state to file slash state.
                state = State::FileSlash;
            }
            // 4. Otherwise, if base is non-null and base’s scheme is "file":
            else if (base_url.has_value() && base_url->m_scheme == "file") {
                // 1. Set url’s host to base’s host, url’s path to a clone of base’s path, and url’s query to base’s query.
                url->m_host = base_url->m_host;
                url->m_paths = base_url->m_paths;
                url->m_query = base_url->m_query;

                // 2. If c is U+003F (?), then set url’s query to the empty string and state to query state.
                if (code_point == '?') {
                    url->m_query = String {};
                    state = State::Query;
                }
                // 3. Otherwise, if c is U+0023 (#), set url’s fragment to the empty string and state to fragment state.
                else if (code_point == '#') {
                    url->m_fragment = String {};
                    state = State::Fragment;
                }
                // 4. Otherwise, if c is not the EOF code point:
                else if (code_point != end_of_file) {
                    // 1. Set url’s query to null.
                    url->m_query = {};

                    // 2. If the code point substring from pointer to the end of input does not start with a Windows drive letter, then shorten url’s path.
                    auto substring_from_pointer = input.substring_view(iterator - input.begin()).as_string();
                    if (!starts_with_windows_drive_letter(substring_from_pointer)) {
                        shorten_urls_path(*url);
                    }
                    // 3. Otherwise:
                    else {
                        // 1. File-invalid-Windows-drive-letter validation error.
                        report_validation_error();

                        // 2. Set url’s path to « ».
                        url->m_paths.clear();
                    }

                    // 4. Set state to path state and decrease pointer by 1.
                    state = State::Path;
                    continue;
                }
            }
            // 5. Otherwise, set state to path state, and decrease pointer by 1.
            else {
                state = State::Path;
                continue;
            }

            break;
        // -> file slash state, https://url.spec.whatwg.org/#file-slash-state
        case State::FileSlash:
            // 1. If c is U+002F (/) or U+005C (\), then:
            if (code_point == '/' || code_point == '\\') {
                // 1. If c is U+005C (\), invalid-reverse-solidus validation error.
                if (code_point == '\\')
                    report_validation_error();

                // 2. Set state to file host state.
                state = State::FileHost;
            }
            // 2. Otherwise:
            else {
                // 1. If base is non-null and base’s scheme is "file", then:
                if (base_url.has_value() && base_url->m_scheme == "file") {
                    // 1. Set url’s host to base’s host.
                    url->m_host = base_url->m_host;

                    // FIXME: The spec does not seem to mention these steps.
                    url->m_paths = base_url->m_paths;
                    url->m_paths.remove(url->m_paths.size() - 1);

                    // 2. If the code point substring from pointer to the end of input does not start with a Windows drive letter and base’s path[0] is a normalized Windows drive letter, then append base’s path[0] to url’s path.
                    auto substring_from_pointer = input.substring_view(iterator - input.begin()).as_string();
                    if (!starts_with_windows_drive_letter(substring_from_pointer) && is_normalized_windows_drive_letter(base_url->m_paths[0]))
                        url->m_paths.append(base_url->m_paths[0]);
                }

                // 2. Set state to path state, and decrease pointer by 1.
                state = State::Path;
                continue;
            }
            break;
        // -> file host state, https://url.spec.whatwg.org/#file-host-state
        case State::FileHost:
            // 1. If c is the EOF code point, U+002F (/), U+005C (\), U+003F (?), or U+0023 (#), then decrease pointer by 1 and then:
            //    NOTE: decreasing the pointer is done at the bottom of this block.
            if (code_point == end_of_file || code_point == '/' || code_point == '\\' || code_point == '?' || code_point == '#') {
                // 1. If state override is not given and buffer is a Windows drive letter, file-invalid-Windows-drive-letter-host validation error, set state to path state.
                if (!state_override.has_value() && is_windows_drive_letter(buffer.string_view())) {
                    report_validation_error();
                    state = State::Path;
                }
                // 2. Otherwise, if buffer is the empty string, then:
                else if (buffer.is_empty()) {
                    // 1. Set url’s host to the empty string.
                    url->m_host = String {};

                    // 2. If state override is given, then return.
                    if (state_override.has_value())
                        return *url;

                    // 3. Set state to path start state.
                    state = State::PathStart;
                }
                // 3. Otherwise, run these steps:
                else {
                    // 1. Let host be the result of host parsing buffer with url is not special.
                    // FIXME: It seems we are not passing through url is not special through here
                    auto host = parse_host(buffer.string_view(), true);

                    // 2. If host is failure, then return failure.
                    if (!host.has_value())
                        return {};

                    // 3. If host is "localhost", then set host to the empty string.
                    if (host.value().has<String>() && host.value().get<String>() == "localhost"sv)
                        host = String {};

                    // 4. Set url’s host to host.
                    url->m_host = host.release_value();

                    // 5. If state override is given, then return.
                    if (state_override.has_value())
                        return *url;

                    // 6. Set buffer to the empty string and state to path start state.
                    buffer.clear();
                    state = State::PathStart;
                }

                // NOTE: Decrement specified at the top of this 'if' statement.
                continue;
            } else {
                buffer.append_code_point(code_point);
            }
            break;
        // -> path start state, https://url.spec.whatwg.org/#path-start-state
        case State::PathStart:
            // 1. If url is special, then:
            if (url->is_special()) {
                // 1. If c is U+005C (\), invalid-reverse-solidus validation error.
                if (code_point == '\\')
                    report_validation_error();

                // 2. Set state to path state.
                state = State::Path;

                // 3. If c is neither U+002F (/) nor U+005C (\), then decrease pointer by 1.
                if (code_point != '/' && code_point != '\\')
                    continue;
            }
            // 2. Otherwise, if state override is not given and c is U+003F (?), set url’s query to the empty string and state to query state.
            else if (!state_override.has_value() && code_point == '?') {
                url->m_query = String {};
                state = State::Query;
            }
            // 3. Otherwise, if state override is not given and c is U+0023 (#), set url’s fragment to the empty string and state to fragment state.
            else if (!state_override.has_value() && code_point == '#') {
                url->m_fragment = String {};
                state = State::Fragment;
            }
            // 4. Otherwise, if c is not the EOF code point:
            else if (code_point != end_of_file) {
                // 1. Set state to path state.
                state = State::Path;

                // 2. If c is not U+002F (/), then decrease pointer by 1.
                if (code_point != '/')
                    continue;
            }
            // 5. Otherwise, if state override is given and url’s host is null, append the empty string to url’s path.
            else if (state_override.has_value() && url->host().has<Empty>()) {
                url->append_slash();
            }
            break;
        // -> path state, https://url.spec.whatwg.org/#path-state
        case State::Path:
            // 1. If one of the following is true:
            //    * c is the EOF code point or U+002F (/)
            //    * url is special and c is U+005C (\)
            //    * state override is not given and c is U+003F (?) or U+0023 (#)
            if ((code_point == end_of_file || code_point == '/')
                || (url->is_special() && code_point == '\\')
                || (!state_override.has_value() && (code_point == '?' || code_point == '#'))) {
                // then:

                // 1. If url is special and c is U+005C (\), invalid-reverse-solidus validation error.
                if (url->is_special() && code_point == '\\')
                    report_validation_error();

                // 2. If buffer is a double-dot URL path segment, then:
                if (is_double_dot_path_segment(buffer.string_view())) {
                    // 1. Shorten url’s path.
                    shorten_urls_path(*url);

                    // 2. If neither c is U+002F (/), nor url is special and c is U+005C (\), append the empty string to url’s path.
                    if (code_point != '/' && !(url->is_special() && code_point == '\\'))
                        url->append_slash();
                }
                // 3. Otherwise, if buffer is a single-dot URL path segment and if neither c is U+002F (/), nor url is special and c is U+005C (\), append the empty string to url’s path.
                else if (is_single_dot_path_segment(buffer.string_view()) && code_point != '/' && !(url->is_special() && code_point == '\\')) {
                    url->append_slash();
                }
                // 4. Otherwise, if buffer is not a single-dot URL path segment, then:
                else if (!is_single_dot_path_segment(buffer.string_view())) {
                    // 1. If url’s scheme is "file", url’s path is empty, and buffer is a Windows drive letter, then replace the second code point in buffer with U+003A (:).
                    if (url->m_scheme == "file" && url->m_paths.is_empty() && is_windows_drive_letter(buffer.string_view())) {
                        auto drive_letter = buffer.string_view()[0];
                        buffer.clear();
                        buffer.append(drive_letter);
                        buffer.append(':');
                    }
                    // 2. Append buffer to url’s path.
                    url->m_paths.append(buffer.to_string_without_validation());
                }

                // 5. Set buffer to the empty string.
                buffer.clear();

                // 6. If c is U+003F (?), then set url’s query to the empty string and state to query state.
                if (code_point == '?') {
                    url->m_query = String {};
                    state = State::Query;
                }
                // 7. If c is U+0023 (#), then set url’s fragment to the empty string and state to fragment state.
                else if (code_point == '#') {
                    url->m_fragment = String {};
                    state = State::Fragment;
                }
            }
            // 2. Otherwise, run these steps
            else {
                // 1. If c is not a URL code point and not U+0025 (%), invalid-URL-unit validation error.
                if (!is_url_code_point(code_point) && code_point != '%')
                    report_validation_error();

                // 2. If c is U+0025 (%) and remaining does not start with two ASCII hex digits, validation error.
                if (code_point == '%' && !remaining_starts_with_two_ascii_hex_digits())
                    report_validation_error();

                // 3. UTF-8 percent-encode c using the path percent-encode set and append the result to buffer.
                URL::append_percent_encoded_if_necessary(buffer, code_point, URL::PercentEncodeSet::Path);
            }
            break;
        // -> opaque path state, https://url.spec.whatwg.org/#cannot-be-a-base-url-path-state
        case State::CannotBeABaseUrlPath:
            // NOTE: This does not follow the spec exactly but rather uses the buffer and only sets the path on EOF.
            VERIFY(url->m_paths.size() == 1 && url->m_paths[0].is_empty());

            // 1. If c is U+003F (?), then set url’s query to the empty string and state to query state.
            if (code_point == '?') {
                url->m_paths[0] = buffer.to_string_without_validation();
                url->m_query = String {};
                buffer.clear();
                state = State::Query;
            }
            // 2. Otherwise, if c is U+0023 (#), then set url’s fragment to the empty string and state to fragment state.
            else if (code_point == '#') {
                // NOTE: This needs to be percent decoded since the member variables contain decoded data.
                url->m_paths[0] = buffer.to_string_without_validation();
                url->m_fragment = String {};
                buffer.clear();
                state = State::Fragment;
            }
            // 3. Otherwise:
            else {
                // 1. If c is not the EOF code point, not a URL code point, and not U+0025 (%), invalid-URL-unit validation error.
                if (code_point != end_of_file && !is_url_code_point(code_point) && code_point != '%')
                    report_validation_error();

                // 2. If c is U+0025 (%) and remaining does not start with two ASCII hex digits, validation error.
                if (code_point == '%' && !remaining_starts_with_two_ascii_hex_digits())
                    report_validation_error();

                // 3. If c is not the EOF code point, UTF-8 percent-encode c using the C0 control percent-encode set and append the result to url’s path.
                if (code_point != end_of_file) {
                    URL::append_percent_encoded_if_necessary(buffer, code_point, URL::PercentEncodeSet::C0Control);
                } else {
                    url->m_paths[0] = buffer.to_string_without_validation();
                    buffer.clear();
                }
            }
            break;
        // -> query state, https://url.spec.whatwg.org/#query-state
        case State::Query:
            // FIXME: 1. If encoding is not UTF-8 and one of the following is true:
            //           * url is not special
            //           * url’s scheme is "ws" or "wss"
            //        then set encoding to UTF-8.

            // 2. If one of the following is true:
            //    * state override is not given and c is U+0023 (#)
            //    * c is the EOF code point
            if ((!state_override.has_value() && code_point == '#')
                || code_point == end_of_file) {
                // then:

                // 1. Let queryPercentEncodeSet be the special-query percent-encode set if url is special; otherwise the query percent-encode set.
                auto query_percent_encode_set = url->is_special() ? URL::PercentEncodeSet::SpecialQuery : URL::PercentEncodeSet::Query;

                // 2. Percent-encode after encoding, with encoding, buffer, and queryPercentEncodeSet, and append the result to url’s query.
                url->m_query = percent_encode_after_encoding(buffer.string_view(), query_percent_encode_set).release_value_but_fixme_should_propagate_errors();

                // 3. Set buffer to the empty string.
                buffer.clear();

                // 4. If c is U+0023 (#), then set url’s fragment to the empty string and state to fragment state.
                if (code_point == '#') {
                    url->m_fragment = String {};
                    state = State::Fragment;
                }
            }
            // 3. Otherwise, if c is not the EOF code point:
            else if (code_point != end_of_file) {
                // 1. If c is not a URL code point and not U+0025 (%), invalid-URL-unit validation error.
                if (!is_url_code_point(code_point) && code_point != '%')
                    report_validation_error();

                // 2. If c is U+0025 (%) and remaining does not start with two ASCII hex digits, validation error.
                if (code_point == '%' && !remaining_starts_with_two_ascii_hex_digits())
                    report_validation_error();

                // 3. Append c to buffer.
                buffer.append_code_point(code_point);
            }
            break;
        // -> fragment state, https://url.spec.whatwg.org/#fragment-state
        case State::Fragment:
            // NOTE: This does not follow the spec exactly but rather uses the buffer and only sets the fragment on EOF.
            // 1. If c is not the EOF code point, then:
            if (code_point != end_of_file) {
                // 1. If c is not a URL code point and not U+0025 (%), invalid-URL-unit validation error.
                if (!is_url_code_point(code_point) && code_point != '%')
                    report_validation_error();

                // 2. If c is U+0025 (%) and remaining does not start with two ASCII hex digits, validation error.
                if (code_point == '%' && !remaining_starts_with_two_ascii_hex_digits())
                    report_validation_error();

                // 3. UTF-8 percent-encode c using the fragment percent-encode set and append the result to url’s fragment.
                // NOTE: The percent-encode is done on EOF on the entire buffer.
                buffer.append_code_point(code_point);
            } else {
                url->m_fragment = percent_encode_after_encoding(buffer.string_view(), URL::PercentEncodeSet::Fragment).release_value_but_fixme_should_propagate_errors();
                buffer.clear();
            }
            break;
        default:
            VERIFY_NOT_REACHED();
        }

        if (iterator.done())
            break;
        ++iterator;
    }

    url->m_valid = true;
