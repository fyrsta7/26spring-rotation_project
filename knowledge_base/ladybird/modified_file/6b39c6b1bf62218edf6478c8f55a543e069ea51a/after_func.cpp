
    return suggestions;
}

void Shell::bring_cursor_to_beginning_of_a_line() const
{
    struct winsize ws;
    if (m_editor) {
        ws = m_editor->terminal_size();
    } else {
        if (ioctl(STDERR_FILENO, TIOCGWINSZ, &ws) < 0) {
            // Very annoying assumptions.
            ws.ws_col = 80;
            ws.ws_row = 25;
        }
    }

    // Black with Cyan background.
    constexpr auto default_mark = "\e[30;46m%\e[0m";
    String eol_mark = getenv("PROMPT_EOL_MARK");
    if (eol_mark.is_null())
        eol_mark = default_mark;
    size_t eol_mark_length = Line::Editor::actual_rendered_string_metrics(eol_mark).line_metrics.last().total_length();
    if (eol_mark_length >= ws.ws_col) {
        eol_mark = default_mark;
        eol_mark_length = 1;
    }

    fputs(eol_mark.characters(), stderr);

    // We write a line's worth of whitespace to the terminal. This way, we ensure that
    // the prompt ends up on a new line even if there is dangling output on the current line.
    size_t fill_count = ws.ws_col - eol_mark_length;
    auto fill_buffer = String::repeated(' ', fill_count);
