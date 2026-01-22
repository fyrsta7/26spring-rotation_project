    std::vector<Task> getInitTasks() {
        return {
            { "Setting up environment",  setupEnvironment,    false },
            { "Creating directories",    createDirectories,   false },
            #if defined(OS_LINUX)
            { "Migrate config to .config", migrateConfig,     false },
            #endif
            { "Loading settings",        loadSettings,        false },
            { "Loading plugins",         loadPlugins,         false },
            { "Checking for updates",    checkForUpdates,     true  },
            { "Loading fonts",           loadFonts,           true  },
        };
    }
