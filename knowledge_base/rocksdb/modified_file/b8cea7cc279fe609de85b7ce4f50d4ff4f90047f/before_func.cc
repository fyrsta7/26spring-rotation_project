  void Apply(VersionEdit* edit) {
    CheckConsistency(base_vstorage_);

    // Delete files
    const VersionEdit::DeletedFileSet& del = edit->GetDeletedFiles();
    for (const auto& del_file : del) {
      const auto level = del_file.first;
      const auto number = del_file.second;
      if (level < num_levels_) {
        levels_[level].deleted_files.insert(number);
        CheckConsistencyForDeletes(edit, number, level);

        auto exising = levels_[level].added_files.find(number);
        if (exising != levels_[level].added_files.end()) {
          UnrefFile(exising->second);
          levels_[level].added_files.erase(number);
        }
      } else {
        if (invalid_levels_[level].count(number) > 0) {
          invalid_levels_[level].erase(number);
        } else {
          // Deleting an non-existing file on invalid level.
          has_invalid_levels_ = true;
        }
      }
    }

    // Add new files
    for (const auto& new_file : edit->GetNewFiles()) {
      const int level = new_file.first;
      if (level < num_levels_) {
        FileMetaData* f = new FileMetaData(new_file.second);
        f->refs = 1;

        assert(levels_[level].added_files.find(f->fd.GetNumber()) ==
               levels_[level].added_files.end());
        levels_[level].deleted_files.erase(f->fd.GetNumber());
        levels_[level].added_files[f->fd.GetNumber()] = f;
      } else {
        uint64_t number = new_file.second.fd.GetNumber();
        if (invalid_levels_[level].count(number) == 0) {
          invalid_levels_[level].insert(number);
        } else {
          // Creating an already existing file on invalid level.
          has_invalid_levels_ = true;
        }
      }
    }
  }
