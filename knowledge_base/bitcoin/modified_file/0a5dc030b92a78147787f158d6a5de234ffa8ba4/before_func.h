    path& operator=(std::filesystem::path path) { std::filesystem::path::operator=(std::move(path)); return *this; }
