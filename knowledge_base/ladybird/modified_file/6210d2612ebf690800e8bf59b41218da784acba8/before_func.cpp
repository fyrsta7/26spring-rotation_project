    m_roots = move(roots);
    m_model->update();
}

Result<NonnullOwnPtr<Profile>, String> Profile::load_from_perfcore_file(const StringView& path)
{
    auto file = Core::File::construct(path);
    if (!file->open(Core::OpenMode::ReadOnly))
        return String::formatted("Unable to open {}, error: {}", path, file->error_string());

    auto json = JsonValue::from_string(file->read_all());
    if (!json.has_value() || !json.value().is_object())
        return String { "Invalid perfcore format (not a JSON object)" };

    auto& object = json.value().as_object();

    auto file_or_error = MappedFile::map("/boot/Kernel");
    OwnPtr<ELF::Image> kernel_elf;
    if (!file_or_error.is_error())
        kernel_elf = make<ELF::Image>(file_or_error.value()->bytes());

    auto events_value = object.get("events");
    if (!events_value.is_array())
        return String { "Malformed profile (events is not an array)" };

    auto& perf_events = events_value.as_array();

    NonnullOwnPtrVector<Process> all_processes;
    HashMap<pid_t, Process*> current_processes;
    Vector<Event> events;

    for (auto& perf_event_value : perf_events.values()) {
        auto& perf_event = perf_event_value.as_object();

        Event event;

        event.timestamp = perf_event.get("timestamp").to_number<u64>();
        event.lost_samples = perf_event.get("lost_samples").to_number<u32>();
        event.type = perf_event.get("type").to_string();
        event.pid = perf_event.get("pid").to_i32();
        event.tid = perf_event.get("tid").to_i32();

        if (event.type == "malloc"sv) {
            event.ptr = perf_event.get("ptr").to_number<FlatPtr>();
            event.size = perf_event.get("size").to_number<size_t>();
        } else if (event.type == "free"sv) {
            event.ptr = perf_event.get("ptr").to_number<FlatPtr>();
        } else if (event.type == "mmap"sv) {
            event.ptr = perf_event.get("ptr").to_number<FlatPtr>();
            event.size = perf_event.get("size").to_number<size_t>();
            event.name = perf_event.get("name").to_string();

            auto it = current_processes.find(event.pid);
            if (it != current_processes.end())
                it->value->library_metadata.handle_mmap(event.ptr, event.size, event.name);
            continue;
        } else if (event.type == "munmap"sv) {
            event.ptr = perf_event.get("ptr").to_number<FlatPtr>();
            event.size = perf_event.get("size").to_number<size_t>();
            continue;
        } else if (event.type == "process_create"sv) {
            event.parent_pid = perf_event.get("parent_pid").to_number<FlatPtr>();
            event.executable = perf_event.get("executable").to_string();

            auto sampled_process = adopt_own(*new Process {
                .pid = event.pid,
                .executable = event.executable,
                .start_valid = event.timestamp,
            });

            current_processes.set(sampled_process->pid, sampled_process);
            all_processes.append(move(sampled_process));
            continue;
        } else if (event.type == "process_exec"sv) {
            event.executable = perf_event.get("executable").to_string();

            auto old_process = current_processes.get(event.pid).value();
            old_process->end_valid = event.timestamp - 1;

            current_processes.remove(event.pid);

            auto sampled_process = adopt_own(*new Process {
                .pid = event.pid,
                .executable = event.executable,
                .start_valid = event.timestamp });

            current_processes.set(sampled_process->pid, sampled_process);
            all_processes.append(move(sampled_process));
            continue;
        } else if (event.type == "process_exit"sv) {
            auto old_process = current_processes.get(event.pid).value();
            old_process->end_valid = event.timestamp - 1;

            current_processes.remove(event.pid);
            continue;
        } else if (event.type == "thread_create"sv) {
            event.parent_tid = perf_event.get("parent_tid").to_i32();
            auto it = current_processes.find(event.pid);
            if (it != current_processes.end())
                it->value->handle_thread_create(event.tid, event.timestamp);
            continue;
        } else if (event.type == "thread_exit"sv) {
            auto it = current_processes.find(event.pid);
            if (it != current_processes.end())
                it->value->handle_thread_exit(event.tid, event.timestamp);
            continue;
        }

        auto stack_array = perf_event.get("stack").as_array();
        for (ssize_t i = stack_array.values().size() - 1; i >= 0; --i) {
            auto& frame = stack_array.at(i);
            auto ptr = frame.to_number<u32>();
            u32 offset = 0;
            FlyString object_name;
            String symbol;

            if (ptr >= 0xc0000000) {
                if (kernel_elf) {
                    symbol = kernel_elf->symbolicate(ptr, &offset);
                } else {
                    symbol = String::formatted("?? <{:p}>", ptr);
                }
            } else {
                auto it = current_processes.find(event.pid);
                // FIXME: This logic is kinda gnarly, find a way to clean it up.
                LibraryMetadata* library_metadata {};
                if (it != current_processes.end())
                    library_metadata = &it->value->library_metadata;
                if (auto* library = library_metadata ? library_metadata->library_containing(ptr) : nullptr) {
                    object_name = library->name;
                    symbol = library->symbolicate(ptr, &offset);
                } else {
                    symbol = String::formatted("?? <{:p}>", ptr);
                }
            }

            event.frames.append({ object_name, symbol, ptr, offset });
        }

        if (event.frames.size() < 2)
            continue;

        FlatPtr innermost_frame_address = event.frames.at(1).address;
        event.in_kernel = innermost_frame_address >= 0xc0000000;

        events.append(move(event));
    }

    if (events.is_empty())
        return String { "No events captured (targeted process was never on CPU)" };

    quick_sort(all_processes, [](auto& a, auto& b) {
        if (a.pid == b.pid)
            return a.start_valid < b.start_valid;
        else
            return a.pid < b.pid;
    });

    Vector<Process> processes;
    for (auto& it : all_processes)
