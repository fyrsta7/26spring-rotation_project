Status VersionSet::Recover(
    const std::vector<ColumnFamilyDescriptor>& column_families,
    bool read_only) {
  std::unordered_map<std::string, ColumnFamilyOptions> cf_name_to_options;
  for (auto cf : column_families) {
    cf_name_to_options.insert({cf.name, cf.options});
  }
  // keeps track of column families in manifest that were not found in
  // column families parameters. if those column families are not dropped
  // by subsequent manifest records, Recover() will return failure status
  std::unordered_map<int, std::string> column_families_not_found;

  // Read "CURRENT" file, which contains a pointer to the current manifest file
  std::string manifest_filename;
  Status s = ReadFileToString(
      env_, CurrentFileName(dbname_), &manifest_filename
  );
  if (!s.ok()) {
    return s;
  }
  if (manifest_filename.empty() ||
      manifest_filename.back() != '\n') {
    return Status::Corruption("CURRENT file does not end with newline");
  }
  // remove the trailing '\n'
  manifest_filename.resize(manifest_filename.size() - 1);
  FileType type;
  bool parse_ok =
      ParseFileName(manifest_filename, &manifest_file_number_, &type);
  if (!parse_ok || type != kDescriptorFile) {
    return Status::Corruption("CURRENT file corrupted");
  }

  Log(db_options_->info_log, "Recovering from manifest file: %s\n",
      manifest_filename.c_str());

  manifest_filename = dbname_ + "/" + manifest_filename;
  unique_ptr<SequentialFile> manifest_file;
  s = env_->NewSequentialFile(manifest_filename, &manifest_file,
                              env_options_);
  if (!s.ok()) {
    return s;
  }
  uint64_t manifest_file_size;
  s = env_->GetFileSize(manifest_filename, &manifest_file_size);
  if (!s.ok()) {
    return s;
  }

  bool have_log_number = false;
  bool have_prev_log_number = false;
  bool have_next_file = false;
  bool have_last_sequence = false;
  uint64_t next_file = 0;
  uint64_t last_sequence = 0;
  uint64_t log_number = 0;
  uint64_t prev_log_number = 0;
  uint32_t max_column_family = 0;
  std::unordered_map<uint32_t, Builder*> builders;

  // add default column family
  auto default_cf_iter = cf_name_to_options.find(kDefaultColumnFamilyName);
  if (default_cf_iter == cf_name_to_options.end()) {
    return Status::InvalidArgument("Default column family not specified");
  }
  VersionEdit default_cf_edit;
  default_cf_edit.AddColumnFamily(kDefaultColumnFamilyName);
  default_cf_edit.SetColumnFamily(0);
  ColumnFamilyData* default_cfd =
      CreateColumnFamily(default_cf_iter->second, &default_cf_edit);
  builders.insert({0, new Builder(default_cfd)});

  {
    VersionSet::LogReporter reporter;
    reporter.status = &s;
    log::Reader reader(std::move(manifest_file), &reporter, true /*checksum*/,
                       0 /*initial_offset*/);
    Slice record;
    std::string scratch;
    while (reader.ReadRecord(&record, &scratch) && s.ok()) {
      VersionEdit edit;
      s = edit.DecodeFrom(record);
      if (!s.ok()) {
        break;
      }

      // Not found means that user didn't supply that column
      // family option AND we encountered column family add
      // record. Once we encounter column family drop record,
      // we will delete the column family from
      // column_families_not_found.
      bool cf_in_not_found =
          column_families_not_found.find(edit.column_family_) !=
          column_families_not_found.end();
      // in builders means that user supplied that column family
      // option AND that we encountered column family add record
      bool cf_in_builders =
          builders.find(edit.column_family_) != builders.end();

      // they can't both be true
      assert(!(cf_in_not_found && cf_in_builders));

      ColumnFamilyData* cfd = nullptr;

      if (edit.is_column_family_add_) {
        if (cf_in_builders || cf_in_not_found) {
          s = Status::Corruption(
              "Manifest adding the same column family twice");
          break;
        }
        auto cf_options = cf_name_to_options.find(edit.column_family_name_);
        if (cf_options == cf_name_to_options.end()) {
          column_families_not_found.insert(
              {edit.column_family_, edit.column_family_name_});
        } else {
          cfd = CreateColumnFamily(cf_options->second, &edit);
          builders.insert({edit.column_family_, new Builder(cfd)});
        }
      } else if (edit.is_column_family_drop_) {
        if (cf_in_builders) {
          auto builder = builders.find(edit.column_family_);
          assert(builder != builders.end());
          delete builder->second;
          builders.erase(builder);
          cfd = column_family_set_->GetColumnFamily(edit.column_family_);
          if (cfd->Unref()) {
            delete cfd;
            cfd = nullptr;
          } else {
            // who else can have reference to cfd!?
            assert(false);
          }
        } else if (cf_in_not_found) {
          column_families_not_found.erase(edit.column_family_);
        } else {
          s = Status::Corruption(
              "Manifest - dropping non-existing column family");
          break;
        }
      } else if (!cf_in_not_found) {
        if (!cf_in_builders) {
          s = Status::Corruption(
              "Manifest record referencing unknown column family");
          break;
        }

        cfd = column_family_set_->GetColumnFamily(edit.column_family_);
        // this should never happen since cf_in_builders is true
        assert(cfd != nullptr);
        if (edit.max_level_ >= cfd->current()->NumberLevels()) {
          s = Status::InvalidArgument(
              "db has more levels than options.num_levels");
          break;
        }

        // if it is not column family add or column family drop,
        // then it's a file add/delete, which should be forwarded
        // to builder
        auto builder = builders.find(edit.column_family_);
        assert(builder != builders.end());
        builder->second->Apply(&edit);
      }

      if (cfd != nullptr) {
        if (edit.has_log_number_) {
          if (cfd->GetLogNumber() > edit.log_number_) {
            Log(db_options_->info_log,
                "MANIFEST corruption detected, but ignored - Log numbers in "
                "records NOT monotonically increasing");
          } else {
            cfd->SetLogNumber(edit.log_number_);
            have_log_number = true;
          }
        }
        if (edit.has_comparator_ &&
            edit.comparator_ != cfd->user_comparator()->Name()) {
          s = Status::InvalidArgument(
              cfd->user_comparator()->Name(),
              "does not match existing comparator " + edit.comparator_);
          break;
        }
      }

      if (edit.has_prev_log_number_) {
        prev_log_number = edit.prev_log_number_;
        have_prev_log_number = true;
      }

      if (edit.has_next_file_number_) {
        next_file = edit.next_file_number_;
        have_next_file = true;
      }

      if (edit.has_max_column_family_) {
        max_column_family = edit.max_column_family_;
      }

      if (edit.has_last_sequence_) {
        last_sequence = edit.last_sequence_;
        have_last_sequence = true;
      }
    }
  }

  if (s.ok()) {
    if (!have_next_file) {
      s = Status::Corruption("no meta-nextfile entry in descriptor");
    } else if (!have_log_number) {
      s = Status::Corruption("no meta-lognumber entry in descriptor");
    } else if (!have_last_sequence) {
      s = Status::Corruption("no last-sequence-number entry in descriptor");
    }

    if (!have_prev_log_number) {
      prev_log_number = 0;
    }

    column_family_set_->UpdateMaxColumnFamily(max_column_family);

    MarkFileNumberUsed(prev_log_number);
    MarkFileNumberUsed(log_number);
  }

  // there were some column families in the MANIFEST that weren't specified
  // in the argument. This is OK in read_only mode
  if (read_only == false && column_families_not_found.size() > 0) {
    std::string list_of_not_found;
    for (const auto& cf : column_families_not_found) {
      list_of_not_found += ", " + cf.second;
    }
    list_of_not_found = list_of_not_found.substr(2);
    s = Status::InvalidArgument(
        "You have to open all column families. Column families not opened: " +
        list_of_not_found);
  }

  if (s.ok()) {
    for (auto cfd : *column_family_set_) {
      auto builders_iter = builders.find(cfd->GetID());
      assert(builders_iter != builders.end());
      auto builder = builders_iter->second;

      if (db_options_->max_open_files == -1) {
      // unlimited table cache. Pre-load table handle now.
      // Need to do it out of the mutex.
        builder->LoadTableHandlers();
      }

      Version* v = new Version(cfd, this, current_version_number_++);
      builder->SaveTo(v);

      // Install recovered version
      std::vector<uint64_t> size_being_compacted(v->NumberLevels() - 1);
      cfd->compaction_picker()->SizeBeingCompacted(size_being_compacted);
      v->PrepareApply(size_being_compacted);
      AppendVersion(cfd, v);
    }

    manifest_file_size_ = manifest_file_size;
    next_file_number_ = next_file + 1;
    last_sequence_ = last_sequence;
    prev_log_number_ = prev_log_number;

    Log(db_options_->info_log,
        "Recovered from manifest file:%s succeeded,"
        "manifest_file_number is %lu, next_file_number is %lu, "
        "last_sequence is %lu, log_number is %lu,"
        "prev_log_number is %lu,"
        "max_column_family is %u\n",
        manifest_filename.c_str(), (unsigned long)manifest_file_number_,
        (unsigned long)next_file_number_, (unsigned long)last_sequence_,
        (unsigned long)log_number, (unsigned long)prev_log_number_,
        column_family_set_->GetMaxColumnFamily());

    for (auto cfd : *column_family_set_) {
      Log(db_options_->info_log,
          "Column family [%s] (ID %u), log number is %" PRIu64 "\n",
          cfd->GetName().c_str(), cfd->GetID(), cfd->GetLogNumber());
    }
  }

  for (auto builder : builders) {
    delete builder.second;
  }

  return s;
}