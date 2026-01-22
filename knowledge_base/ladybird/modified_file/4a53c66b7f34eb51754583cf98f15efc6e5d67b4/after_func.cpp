
    return window;
}

void ClientConnection::request_file_handler(i32 window_server_client_id, i32 parent_window_id, String const& path, Core::OpenMode const& requested_access, ShouldPrompt prompt)
{
    VERIFY(path.starts_with("/"sv));

    bool approved = false;
    auto maybe_permissions = m_approved_files.get(path);

    auto relevant_permissions = requested_access & (Core::OpenMode::ReadOnly | Core::OpenMode::WriteOnly);
    VERIFY(relevant_permissions != Core::OpenMode::NotOpen);

    if (maybe_permissions.has_value())
        approved = has_flag(maybe_permissions.value(), relevant_permissions);

    if (!approved) {
        String access_string;

        if (has_flag(requested_access, Core::OpenMode::ReadWrite))
            access_string = "read and write";
        else if (has_flag(requested_access, Core::OpenMode::ReadOnly))
            access_string = "read from";
        else if (has_flag(requested_access, Core::OpenMode::WriteOnly))
            access_string = "write to";

        auto pid = this->socket().peer_pid();
        auto exe_link = LexicalPath("/proc").append(String::number(pid)).append("exe").string();
        auto exe_path = Core::File::real_path_for(exe_link);

        auto main_window = create_dummy_child_window(window_server_client_id, parent_window_id);

        if (prompt == ShouldPrompt::Yes) {
            auto exe_name = LexicalPath::basename(exe_path);
            auto result = GUI::MessageBox::show(main_window, String::formatted("Allow {} ({}) to {} \"{}\"?", exe_name, pid, access_string, path), "File Permissions Requested", GUI::MessageBox::Type::Warning, GUI::MessageBox::InputType::YesNo);
            approved = result == GUI::MessageBox::ExecYes;
        } else {
            approved = true;
        }

        if (approved) {
            auto new_permissions = relevant_permissions;

            if (maybe_permissions.has_value())
                new_permissions |= maybe_permissions.value();

            m_approved_files.set(path, new_permissions);
        }
    }

    if (approved) {
        auto file = Core::File::open(path, requested_access);

        if (file.is_error()) {
            dbgln("FileSystemAccessServer: Couldn't open {}, error {}", path, file.error());
            async_handle_prompt_end(errno, Optional<IPC::File> {}, path);
        } else {
            async_handle_prompt_end(0, IPC::File(file.value()->leak_fd(), IPC::File::CloseAfterSending), path);
        }
