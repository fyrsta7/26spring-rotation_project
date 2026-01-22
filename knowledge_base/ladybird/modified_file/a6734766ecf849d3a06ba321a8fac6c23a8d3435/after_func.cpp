}

void Terminal::escape$P(const ParamVector& params)
{
    int num = 1;
    if (params.size() >= 1)
        num = params[0];

    if (num == 0)
        num = 1;

    auto& line = m_lines[m_cursor_row];

    // Move n characters of line to the left
    for (int i = m_cursor_column; i < line.length() - num; i++)
        line.set_code_point(i, line.code_point(i + num));

    // Fill remainder of line with blanks
    for (int i = line.length() - num; i < line.length(); i++)
        line.set_code_point(i, ' ');

    line.set_dirty(true);
}

void Terminal::execute_xterm_command()
{
    ParamVector numeric_params;
    auto param_string = String::copy(m_xterm_parameters);
    auto params = param_string.split(';', true);
    m_xterm_parameters.clear_with_capacity();
    for (auto& parampart : params)
        numeric_params.append(parampart.to_uint().value_or(0));

    while (params.size() < 3) {
        params.append(String::empty());
        numeric_params.append(0);
    }

    m_final = '@';

    if (numeric_params.is_empty()) {
        dbg() << "Empty Xterm params?";
        return;
    }

