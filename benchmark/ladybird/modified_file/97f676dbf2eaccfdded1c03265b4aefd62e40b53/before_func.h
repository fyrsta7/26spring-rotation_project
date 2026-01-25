    void push_command(PaintingCommand command)
    {
        m_painting_commands.append({ state().scroll_frame_id, command });
    }