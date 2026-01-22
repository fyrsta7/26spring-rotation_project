    m_resolutions.append({ 1360, 768 });
    m_resolutions.append({ 1368, 768 });
    m_resolutions.append({ 1440, 900 });
    m_resolutions.append({ 1600, 900 });
    m_resolutions.append({ 1920, 1080 });
    m_resolutions.append({ 2560, 1080 });
}

void DisplaySettingsWidget::create_wallpaper_list()
{
    Core::DirIterator iterator("/res/wallpapers/", Core::DirIterator::Flags::SkipDots);

    m_wallpapers.append("Use background color");

    while (iterator.has_next()) {
        m_wallpapers.append(iterator.next_path());
    }

    m_modes.append("simple");
    m_modes.append("tile");
    m_modes.append("center");
    m_modes.append("scaled");
}

void DisplaySettingsWidget::create_frame()
{
    m_root_widget = GUI::Widget::construct();
    m_root_widget->set_layout<GUI::VerticalBoxLayout>();
    m_root_widget->set_fill_with_background_color(true);
    m_root_widget->layout()->set_margins({ 4, 4, 4, 4 });

    auto& settings_content = m_root_widget->add<GUI::Widget>();
    settings_content.set_layout<GUI::VerticalBoxLayout>();
    settings_content.set_backcolor("red");
    settings_content.set_background_color(Color::Blue);
    settings_content.set_background_role(Gfx::ColorRole::Window);
    settings_content.layout()->set_margins({ 4, 4, 4, 4 });

    /// Wallpaper Preview /////////////////////////////////////////////////////////////////////////

    m_monitor_widget = settings_content.add<MonitorWidget>();
    m_monitor_widget->set_size_policy(GUI::SizePolicy::Fixed, GUI::SizePolicy::Fixed);
    m_monitor_widget->set_preferred_size(338, 248);

    /// Wallpaper Row /////////////////////////////////////////////////////////////////////////////

    auto& wallpaper_selection_container = settings_content.add<GUI::Widget>();
    wallpaper_selection_container.set_layout<GUI::HorizontalBoxLayout>();
    wallpaper_selection_container.layout()->set_margins({ 0, 4, 0, 0 });
    wallpaper_selection_container.set_size_policy(GUI::SizePolicy::Fill, GUI::SizePolicy::Fixed);
    wallpaper_selection_container.set_preferred_size(0, 22);

    auto& wallpaper_label = wallpaper_selection_container.add<GUI::Label>();
    wallpaper_label.set_text_alignment(Gfx::TextAlignment::CenterLeft);
    wallpaper_label.set_size_policy(GUI::SizePolicy::Fixed, GUI::SizePolicy::Fill);
    wallpaper_label.set_preferred_size({ 70, 0 });
    wallpaper_label.set_text("Wallpaper:");

    m_wallpaper_combo = wallpaper_selection_container.add<GUI::ComboBox>();
    m_wallpaper_combo->set_size_policy(GUI::SizePolicy::Fill, GUI::SizePolicy::Fixed);
    m_wallpaper_combo->set_preferred_size(0, 22);
    m_wallpaper_combo->set_only_allow_values_from_model(true);
    m_wallpaper_combo->set_model(*GUI::ItemListModel<AK::String>::create(m_wallpapers));
    m_wallpaper_combo->on_change = [this](auto& text, const GUI::ModelIndex& index) {
        String path = text;
        if (path.starts_with("/") && m_monitor_widget->set_wallpaper(path)) {
            m_monitor_widget->update();
            return;
        }

        if (index.row() == 0) {
            path = "";
        } else {
            if (index.is_valid()) {
                StringBuilder builder;
                builder.append("/res/wallpapers/");
                builder.append(path);
                path = builder.to_string();
            }
        }

        m_monitor_widget->set_wallpaper(path);
        m_monitor_widget->update();
    };

    auto& button = wallpaper_selection_container.add<GUI::Button>();
    button.set_tooltip("Select Wallpaper from file system.");
    button.set_icon(Gfx::Bitmap::load_from_file("/res/icons/16x16/open.png"));
    button.set_button_style(Gfx::ButtonStyle::CoolBar);
    button.set_size_policy(GUI::SizePolicy::Fixed, GUI::SizePolicy::Fixed);
    button.set_preferred_size(22, 22);
    button.on_click = [this](auto) {
        Optional<String> open_path = GUI::FilePicker::get_open_filepath(root_widget()->window(), "Select wallpaper from file system.");

        if (!open_path.has_value())
            return;

        m_wallpaper_combo->set_only_allow_values_from_model(false);
        this->m_wallpaper_combo->set_text(open_path.value());
        m_wallpaper_combo->set_only_allow_values_from_model(true);
    };

    /// Mode //////////////////////////////////////////////////////////////////////////////////////

    auto& mode_selection_container = settings_content.add<GUI::Widget>();
    mode_selection_container.set_layout<GUI::HorizontalBoxLayout>();
    mode_selection_container.layout()->set_margins({ 0, 4, 0, 0 });
    mode_selection_container.set_size_policy(GUI::SizePolicy::Fill, GUI::SizePolicy::Fixed);
    mode_selection_container.set_preferred_size(0, 22);

    auto& mode_label = mode_selection_container.add<GUI::Label>();
    mode_label.set_text_alignment(Gfx::TextAlignment::CenterLeft);
    mode_label.set_size_policy(GUI::SizePolicy::Fixed, GUI::SizePolicy::Fill);
    mode_label.set_preferred_size({ 70, 0 });
    mode_label.set_text("Mode:");

    m_mode_combo = mode_selection_container.add<GUI::ComboBox>();
    m_mode_combo->set_size_policy(GUI::SizePolicy::Fill, GUI::SizePolicy::Fixed);
    m_mode_combo->set_preferred_size(0, 22);
    m_mode_combo->set_only_allow_values_from_model(true);
    m_mode_combo->set_model(*GUI::ItemListModel<AK::String>::create(m_modes));
    m_mode_combo->on_change = [this](auto&, const GUI::ModelIndex& index) {
        this->m_monitor_widget->set_wallpaper_mode(m_modes.at(index.row()));
        this->m_monitor_widget->update();
    };

    /// Resolution Row ////////////////////////////////////////////////////////////////////////////

    auto& resolution_selection_container = settings_content.add<GUI::Widget>();
    resolution_selection_container.set_layout<GUI::HorizontalBoxLayout>();
    resolution_selection_container.set_size_policy(GUI::SizePolicy::Fill, GUI::SizePolicy::Fixed);
    resolution_selection_container.set_preferred_size(0, 22);

    auto& m_resolution_label = resolution_selection_container.add<GUI::Label>();
    m_resolution_label.set_text_alignment(Gfx::TextAlignment::CenterLeft);
    m_resolution_label.set_size_policy(GUI::SizePolicy::Fixed, GUI::SizePolicy::Fill);
    m_resolution_label.set_preferred_size({ 70, 0 });
    m_resolution_label.set_text("Resolution:");

    m_resolution_combo = resolution_selection_container.add<GUI::ComboBox>();
    m_resolution_combo->set_size_policy(GUI::SizePolicy::Fill, GUI::SizePolicy::Fixed);
    m_resolution_combo->set_preferred_size(0, 22);
    m_resolution_combo->set_only_allow_values_from_model(true);
    m_resolution_combo->set_model(*GUI::ItemListModel<Gfx::IntSize>::create(m_resolutions));
    m_resolution_combo->on_change = [this](auto&, const GUI::ModelIndex& index) {
        this->m_monitor_widget->set_desktop_resolution(m_resolutions.at(index.row()));
        this->m_monitor_widget->update();
    };

    /// Background Color Row //////////////////////////////////////////////////////////////////////

    auto& color_selection_container = settings_content.add<GUI::Widget>();
    color_selection_container.set_layout<GUI::HorizontalBoxLayout>();
    color_selection_container.set_size_policy(GUI::SizePolicy::Fill, GUI::SizePolicy::Fixed);
    color_selection_container.set_preferred_size(0, 22);

    auto& color_label = color_selection_container.add<GUI::Label>();
    color_label.set_text_alignment(Gfx::TextAlignment::CenterLeft);
    color_label.set_size_policy(GUI::SizePolicy::Fixed, GUI::SizePolicy::Fill);
    color_label.set_preferred_size({ 70, 0 });
    color_label.set_text("Color:");

    m_color_input = color_selection_container.add<GUI::ColorInput>();
    m_color_input->set_color_has_alpha_channel(false);
    m_color_input->set_size_policy(GUI::SizePolicy::Fixed, GUI::SizePolicy::Fill);
    m_color_input->set_preferred_size(90, 0);
    m_color_input->set_color_picker_title("Select color for desktop");
    m_color_input->on_change = [this] {
        this->m_monitor_widget->set_background_color(m_color_input->color());
        this->m_monitor_widget->update();
    };

    /// Add the apply and cancel buttons //////////////////////////////////////////////////////////

    auto& bottom_widget = settings_content.add<GUI::Widget>();
    bottom_widget.set_layout<GUI::HorizontalBoxLayout>();
    bottom_widget.layout()->add_spacer();
    //bottom_widget.layout()->set_margins({ 4, 10, 4, 10 });
    bottom_widget.set_size_policy(Orientation::Vertical, GUI::SizePolicy::Fixed);
    bottom_widget.set_preferred_size(1, 22);

    auto& ok_button = bottom_widget.add<GUI::Button>();
