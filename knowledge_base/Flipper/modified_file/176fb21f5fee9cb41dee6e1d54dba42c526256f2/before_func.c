static void storage_cli_write_chunk(Cli* cli, FuriString* path, FuriString* args) {
    Storage* api = furi_record_open(RECORD_STORAGE);
    File* file = storage_file_alloc(api);

    uint32_t buffer_size;
    int parsed_count = sscanf(furi_string_get_cstr(args), "%lu", &buffer_size);

    if(parsed_count != 1) {
        storage_cli_print_usage();
    } else {
        if(storage_file_open(file, furi_string_get_cstr(path), FSAM_WRITE, FSOM_OPEN_APPEND)) {
            printf("Ready\r\n");

            if(buffer_size) {
                uint8_t* buffer = malloc(buffer_size);

                for(uint32_t i = 0; i < buffer_size; i++) {
                    buffer[i] = cli_getc(cli);
                }

                uint16_t written_size = storage_file_write(file, buffer, buffer_size);

                if(written_size != buffer_size) {
                    storage_cli_print_error(storage_file_get_error(file));
                }

                free(buffer);
            }
        } else {
            storage_cli_print_error(storage_file_get_error(file));
        }
        storage_file_close(file);
    }

    storage_file_free(file);
    furi_record_close(RECORD_STORAGE);
}
