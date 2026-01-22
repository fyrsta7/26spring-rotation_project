static inline ssize_t health_alarm_log_read(RRDHOST *host, FILE *fp, const char *filename) {
    static uint32_t max_unique_id = 0, max_alarm_id = 0;
    ssize_t loaded = -1, updated = -1, errored = -1, duplicate = -1;

    errno = 0;

    char *s, *buf = mallocz(65536 + 1);
    size_t line = 0, len = 0;
    loaded = updated = errored = duplicate = 0;

    pthread_rwlock_rdlock(&host->health_log.alarm_log_rwlock);

    while((s = fgets_trim_len(buf, 65536, fp, &len))) {
        health.log_entries_written++;
        line++;

        int max_entries = 30, entries = 0;
        char *pointers[max_entries];

        pointers[entries++] = s++;
        while(*s) {
            if(unlikely(*s == '\t')) {
                *s = '\0';
                pointers[entries++] = ++s;
                if(entries >= max_entries) {
                    error("Health: line %zu of file '%s' has more than %d entries. Ignoring excessive entries.", line, filename, max_entries);
                    break;
                }
            }
            else s++;
        }

        if(likely(*pointers[0] == 'U' || *pointers[0] == 'A')) {
            ALARM_ENTRY *ae = NULL;

            if(entries < 26) {
                error("Health: line %zu of file '%s' should have at least 26 entries, but it has %d. Ignoring line.", line, filename, entries);
                errored++;
                continue;
            }

            // check that we have valid ids
            uint32_t unique_id = (uint32_t)strtoul(pointers[2], NULL, 16);
            if(!unique_id) {
                error("Health: line %zu of file '%s' states alarm entry with unique id %u (%s). Ignoring line.", line, filename, unique_id, pointers[2]);
                errored++;
                continue;
            }

            uint32_t alarm_id = (uint32_t)strtoul(pointers[3], NULL, 16);
            if(!alarm_id) {
                error("Health: line %zu of file '%s' states alarm entry for alarm id %u (%s). Ignoring line.", line, filename, alarm_id, pointers[3]);
                errored++;
                continue;
            }

            // find a possible overwrite
            for(ae = host->health_log.alarms; ae; ae = ae->next) {
                if(unlikely(ae->unique_id == unique_id)) {
                    if(unlikely(*pointers[0] == 'A')) {
                        error("Health: line %zu of file '%s' adds duplicate alarm log entry with unique id %u."
                              , line, filename, unique_id);
                        *pointers[0] = 'U';
                        duplicate++;
                    }
                    break;
                }
            }

            // if not found, create a new one
            if(likely(!ae)) {

                // if it is an update, but we haven't found it, make it an addition
                if(unlikely(*pointers[0] == 'U')) {
                    *pointers[0] = 'A';
                    error("Health: line %zu of file '%s' updates alarm log entry with unique id %u, but it is not found.", line, filename, unique_id);
                }

                // alarms should be added in the right order
                if(unlikely(unique_id < max_unique_id)) {
                    error("Health: line %zu of file '%s' has alarm log entry with %u in wrong order.", line, filename, unique_id);
                }

                ae = callocz(1, sizeof(ALARM_ENTRY));
            }

            // check for a possible host missmatch
            if(strcmp(pointers[1], host->hostname))
                error("Health: line %zu of file '%s' provides an alarm for host '%s' but this is named '%s'.", line, filename, pointers[1], host->hostname);

            ae->unique_id               = unique_id;
            ae->alarm_id                = alarm_id;
            ae->alarm_event_id          = (uint32_t)strtoul(pointers[4], NULL, 16);
            ae->updated_by_id           = (uint32_t)strtoul(pointers[5], NULL, 16);
            ae->updates_id              = (uint32_t)strtoul(pointers[6], NULL, 16);

            ae->when                    = (uint32_t)strtoul(pointers[7], NULL, 16);
            ae->duration                = (uint32_t)strtoul(pointers[8], NULL, 16);
            ae->non_clear_duration      = (uint32_t)strtoul(pointers[9], NULL, 16);

            ae->flags                   = (uint32_t)strtoul(pointers[10], NULL, 16);
            ae->flags |= HEALTH_ENTRY_FLAG_SAVED;

            ae->exec_run_timestamp      = (uint32_t)strtoul(pointers[11], NULL, 16);
            ae->delay_up_to_timestamp   = (uint32_t)strtoul(pointers[12], NULL, 16);

            if(unlikely(ae->name)) freez(ae->name);
            ae->name = strdupz(pointers[13]);

            if(unlikely(ae->chart)) freez(ae->chart);
            ae->chart = strdupz(pointers[14]);

            if(unlikely(ae->family)) freez(ae->family);
            ae->family = strdupz(pointers[15]);

            if(unlikely(ae->exec)) freez(ae->exec);
            ae->exec = strdupz(pointers[16]);
            if(!*ae->exec) { freez(ae->exec); ae->exec = NULL; }

            if(unlikely(ae->recipient)) freez(ae->recipient);
            ae->recipient = strdupz(pointers[17]);
            if(!*ae->recipient) { freez(ae->recipient); ae->recipient = NULL; }

            if(unlikely(ae->source)) freez(ae->source);
            ae->source = strdupz(pointers[18]);
            if(!*ae->source) { freez(ae->source); ae->source = NULL; }

            if(unlikely(ae->units)) freez(ae->units);
            ae->units = strdupz(pointers[19]);
            if(!*ae->units) { freez(ae->units); ae->units = NULL; }

            if(unlikely(ae->info)) freez(ae->info);
            ae->info = strdupz(pointers[20]);
            if(!*ae->info) { freez(ae->info); ae->info = NULL; }

            ae->exec_code   = atoi(pointers[21]);
            ae->new_status  = atoi(pointers[22]);
            ae->old_status  = atoi(pointers[23]);
            ae->delay       = atoi(pointers[24]);

            ae->new_value   = strtold(pointers[25], NULL);
            ae->old_value   = strtold(pointers[26], NULL);

            // add it to host if not already there
            if(unlikely(*pointers[0] == 'A')) {
                ae->next = host->health_log.alarms;
                host->health_log.alarms = ae;
                loaded++;
            }
            else updated++;

            if(unlikely(ae->unique_id > max_unique_id))
                max_unique_id = ae->unique_id;

            if(unlikely(ae->alarm_id >= max_alarm_id))
                max_alarm_id = ae->alarm_id;
        }
        else {
            error("Health: line %zu of file '%s' is invalid (unrecognized entry type '%s').", line, filename, pointers[0]);
            errored++;
        }
    }

    pthread_rwlock_unlock(&host->health_log.alarm_log_rwlock);

    freez(buf);

    if(!max_unique_id) max_unique_id = (uint32_t)time(NULL);
    if(!max_alarm_id)  max_alarm_id  = (uint32_t)time(NULL);

    host->health_log.next_log_id = max_unique_id + 1;
    host->health_log.next_alarm_id = max_alarm_id + 1;

    info("Health: loaded file '%s' with %zd new alarm entries, updated %zd alarms, errors %zd entries, duplicate %zd", filename, loaded, updated, errored, duplicate);
    return loaded;
}
