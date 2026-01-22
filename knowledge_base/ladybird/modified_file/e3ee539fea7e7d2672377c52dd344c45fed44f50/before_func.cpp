using namespace Profiler;

static bool generate_profile(pid_t& pid);

int main(int argc, char** argv)
{
    int pid = 0;
    const char* perfcore_file_arg = nullptr;
    Core::ArgsParser args_parser;
    args_parser.add_option(pid, "PID to profile", "pid", 'p', "PID");
    args_parser.add_positional_argument(perfcore_file_arg, "Path of perfcore file", "perfcore-file", Core::ArgsParser::Required::No);
    args_parser.parse(argc, argv);

    if (pid && perfcore_file_arg) {
        warnln("-p/--pid option and perfcore-file argument must not be used together!");
        return 1;
    }

    auto app = GUI::Application::construct(argc, argv);
    auto app_icon = GUI::Icon::default_icon("app-profiler");

    String perfcore_file;
    if (!perfcore_file_arg) {
        if (!generate_profile(pid))
            return 0;
        perfcore_file = String::formatted("/proc/{}/perf_events", pid);
    } else {
        perfcore_file = perfcore_file_arg;
    }

    auto profile_or_error = Profile::load_from_perfcore_file(perfcore_file);
    if (profile_or_error.is_error()) {
        GUI::MessageBox::show(nullptr, profile_or_error.error(), "Profiler", GUI::MessageBox::Type::Error);
        return 0;
    }

    auto& profile = profile_or_error.value();

    auto window = GUI::Window::construct();

    if (!Desktop::Launcher::add_allowed_handler_with_only_specific_urls(
            "/bin/Help",
            { URL::create_with_file_protocol("/usr/share/man/man1/Profiler.md") })
        || !Desktop::Launcher::seal_allowlist()) {
        warnln("Failed to set up allowed launch URLs");
        return 1;
    }

    window->set_title("Profiler");
    window->set_icon(app_icon.bitmap_for_size(16));
    window->resize(800, 600);

    auto& main_widget = window->set_main_widget<GUI::Widget>();
    main_widget.set_fill_with_background_color(true);
    main_widget.set_layout<GUI::VerticalBoxLayout>();

    auto timeline_header_container = GUI::Widget::construct();
    timeline_header_container->set_layout<GUI::VerticalBoxLayout>();
    timeline_header_container->set_fill_with_background_color(true);
    timeline_header_container->set_shrink_to_fit(true);

    auto timeline_view = TimelineView::construct();
    for (auto& process : profile->processes()) {
        size_t event_count = 0;
        for (auto& event : profile->events()) {
            if (event.pid == process.pid && process.valid_at(event.timestamp))
                ++event_count;
        }
        if (!event_count)
            continue;
        auto& timeline_header = timeline_header_container->add<TimelineHeader>(*profile, process);
        timeline_header.set_shrink_to_fit(true);
        timeline_header.on_selection_change = [&](bool selected) {
            if (selected) {
                auto end_valid = process.end_valid == 0 ? profile->last_timestamp() : process.end_valid;
                profile->set_process_filter(process.pid, process.start_valid, end_valid);
            } else
                profile->clear_process_filter();

            timeline_header_container->for_each_child_widget([](auto& other_timeline_header) {
                static_cast<TimelineHeader&>(other_timeline_header).update_selection();
                return IterationDecision::Continue;
            });
        };
        timeline_view->add<TimelineTrack>(*timeline_view, *profile, process);
    }

    [[maybe_unused]] auto& timeline_container = main_widget.add<TimelineContainer>(*timeline_header_container, *timeline_view);

    auto& tab_widget = main_widget.add<GUI::TabWidget>();

    auto& tree_tab = tab_widget.add_tab<GUI::Widget>("Call Tree");
    tree_tab.set_layout<GUI::VerticalBoxLayout>();
    tree_tab.layout()->set_margins({ 4, 4, 4, 4 });
    auto& bottom_splitter = tree_tab.add<GUI::VerticalSplitter>();

    auto& tree_view = bottom_splitter.add<GUI::TreeView>();
    tree_view.set_should_fill_selected_rows(true);
    tree_view.set_column_headers_visible(true);
    tree_view.set_model(profile->model());

    auto& disassembly_view = bottom_splitter.add<GUI::TableView>();

    tree_view.on_selection = [&](auto& index) {
        profile->set_disassembly_index(index);
        disassembly_view.set_model(profile->disassembly_model());
    };

    auto& samples_tab = tab_widget.add_tab<GUI::Widget>("Samples");
    samples_tab.set_layout<GUI::VerticalBoxLayout>();
    samples_tab.layout()->set_margins({ 4, 4, 4, 4 });

    auto& samples_splitter = samples_tab.add<GUI::HorizontalSplitter>();
    auto& samples_table_view = samples_splitter.add<GUI::TableView>();
    samples_table_view.set_model(profile->samples_model());

    auto& individual_sample_view = samples_splitter.add<GUI::TableView>();
    samples_table_view.on_selection = [&](const GUI::ModelIndex& index) {
        auto model = IndividualSampleModel::create(*profile, index.data(GUI::ModelRole::Custom).to_integer<size_t>());
        individual_sample_view.set_model(move(model));
    };

    const u64 start_of_trace = profile->first_timestamp();
    const u64 end_of_trace = start_of_trace + profile->length_in_ms();
    const auto clamp_timestamp = [start_of_trace, end_of_trace](u64 timestamp) -> u64 {
        return min(end_of_trace, max(timestamp, start_of_trace));
    };

    auto& statusbar = main_widget.add<GUI::Statusbar>();
    timeline_view->on_selection_change = [&] {
        auto& view = *timeline_view;
        StringBuilder builder;
        u64 normalized_start_time = clamp_timestamp(min(view.select_start_time(), view.select_end_time()));
        u64 normalized_end_time = clamp_timestamp(max(view.select_start_time(), view.select_end_time()));
        u64 normalized_hover_time = clamp_timestamp(view.hover_time());
        builder.appendff("Time: {} ms", normalized_hover_time - start_of_trace);
        if (normalized_start_time != normalized_end_time) {
            auto start = normalized_start_time - start_of_trace;
            auto end = normalized_end_time - start_of_trace;
            builder.appendff(", Selection: {} - {} ms", start, end);
        }
        statusbar.set_text(builder.to_string());
    };

    auto menubar = GUI::Menubar::construct();
    auto& file_menu = menubar->add_menu("&File");
    file_menu.add_action(GUI::CommonActions::make_quit_action([&](auto&) { app->quit(); }));

    auto& view_menu = menubar->add_menu("&View");

    auto invert_action = GUI::Action::create_checkable("&Invert Tree", { Mod_Ctrl, Key_I }, [&](auto& action) {
        profile->set_inverted(action.is_checked());
    });
    invert_action->set_checked(false);
    view_menu.add_action(invert_action);

    auto top_functions_action = GUI::Action::create_checkable("&Top Functions", { Mod_Ctrl, Key_T }, [&](auto& action) {
        profile->set_show_top_functions(action.is_checked());
    });
    top_functions_action->set_checked(false);
    view_menu.add_action(top_functions_action);

    auto percent_action = GUI::Action::create_checkable("Show &Percentages", { Mod_Ctrl, Key_P }, [&](auto& action) {
        profile->set_show_percentages(action.is_checked());
        tree_view.update();
        disassembly_view.update();
    });
    percent_action->set_checked(false);
    view_menu.add_action(percent_action);

    auto& help_menu = menubar->add_menu("&Help");
    help_menu.add_action(GUI::CommonActions::make_help_action([](auto&) {
        Desktop::Launcher::open(URL::create_with_file_protocol("/usr/share/man/man1/Profiler.md"), "/bin/Help");
    }));
    help_menu.add_action(GUI::CommonActions::make_about_action("Profiler", app_icon, window));

