void UnicharAmbigs::LoadUnicharAmbigs(const UNICHARSET &encoder_set, TFile *ambig_file,
                                      int debug_level, bool use_ambigs_for_adaption,
                                      UNICHARSET *unicharset) {
  int i, j;
  UnicharIdVector *adaption_ambigs_entry;
  if (debug_level) {
    tprintf("Reading ambiguities\n");
  }

  int test_ambig_part_size;
  int replacement_ambig_part_size;
  // The space for buffer is allocated on the heap to avoid
  // GCC frame size warning.
  const int kBufferSize = 10 + 2 * kMaxAmbigStringSize;
  char *buffer = new char[kBufferSize];
  char replacement_string[kMaxAmbigStringSize];
  UNICHAR_ID test_unichar_ids[MAX_AMBIG_SIZE + 1];
  int line_num = 0;
  int type = NOT_AMBIG;

  // Determine the version of the ambigs file.
  int version = 0;
  ASSERT_HOST(ambig_file->FGets(buffer, kBufferSize) != nullptr && buffer[0] != '\0');
  if (*buffer == 'v') {
    version = static_cast<int>(strtol(buffer + 1, nullptr, 10));
    ++line_num;
  } else {
    ambig_file->Rewind();
  }
  while (ambig_file->FGets(buffer, kBufferSize) != nullptr) {
    chomp_string(buffer);
    if (debug_level > 2) {
      tprintf("read line %s\n", buffer);
    }
    ++line_num;
    if (!ParseAmbiguityLine(line_num, version, debug_level, encoder_set, buffer,
                            &test_ambig_part_size, test_unichar_ids, &replacement_ambig_part_size,
                            replacement_string, &type)) {
      continue;
    }
    // Construct AmbigSpec and add it to the appropriate AmbigSpec_LIST.
    auto *ambig_spec = new AmbigSpec();
    if (!InsertIntoTable((type == REPLACE_AMBIG) ? replace_ambigs_ : dang_ambigs_,
                         test_ambig_part_size, test_unichar_ids, replacement_ambig_part_size,
                         replacement_string, type, ambig_spec, unicharset)) {
      continue;
    }

    // Update one_to_one_definite_ambigs_.
    if (test_ambig_part_size == 1 && replacement_ambig_part_size == 1 && type == DEFINITE_AMBIG) {
      if (one_to_one_definite_ambigs_[test_unichar_ids[0]] == nullptr) {
        one_to_one_definite_ambigs_[test_unichar_ids[0]] = new UnicharIdVector();
      }
      one_to_one_definite_ambigs_[test_unichar_ids[0]]->push_back(ambig_spec->correct_ngram_id);
    }
    // Update ambigs_for_adaption_.
    if (use_ambigs_for_adaption) {
      std::vector<UNICHAR_ID> encoding;
      // Silently ignore invalid strings, as before, so it is safe to use a
      // universal ambigs file.
      if (unicharset->encode_string(replacement_string, true, &encoding, nullptr, nullptr)) {
        for (i = 0; i < test_ambig_part_size; ++i) {
          if (ambigs_for_adaption_[test_unichar_ids[i]] == nullptr) {
            ambigs_for_adaption_[test_unichar_ids[i]] = new UnicharIdVector();
          }
          adaption_ambigs_entry = ambigs_for_adaption_[test_unichar_ids[i]];
          for (int id_to_insert : encoding) {
            ASSERT_HOST(id_to_insert != INVALID_UNICHAR_ID);
            // Add the new unichar id to adaption_ambigs_entry (only if the
            // vector does not already contain it) keeping it in sorted order.
            for (j = 0;
                 j < adaption_ambigs_entry->size() && (*adaption_ambigs_entry)[j] > id_to_insert;
                 ++j) {
              ;
            }
            if (j < adaption_ambigs_entry->size()) {
              if ((*adaption_ambigs_entry)[j] != id_to_insert) {
                adaption_ambigs_entry->insert(adaption_ambigs_entry->begin() + j, id_to_insert);
              }
            } else {
              adaption_ambigs_entry->push_back(id_to_insert);
            }
          }
        }
      }
    }
  }
  delete[] buffer;

  // Fill in reverse_ambigs_for_adaption from ambigs_for_adaption vector.
  if (use_ambigs_for_adaption) {
    for (i = 0; i < ambigs_for_adaption_.size(); ++i) {
      adaption_ambigs_entry = ambigs_for_adaption_[i];
      if (adaption_ambigs_entry == nullptr) {
        continue;
      }
      for (j = 0; j < adaption_ambigs_entry->size(); ++j) {
        UNICHAR_ID ambig_id = (*adaption_ambigs_entry)[j];
        if (reverse_ambigs_for_adaption_[ambig_id] == nullptr) {
          reverse_ambigs_for_adaption_[ambig_id] = new UnicharIdVector();
        }
        reverse_ambigs_for_adaption_[ambig_id]->push_back(i);
      }
    }
  }

  // Print what was read from the input file.
  if (debug_level > 1) {
    for (int tbl = 0; tbl < 2; ++tbl) {
      const UnicharAmbigsVector &print_table = (tbl == 0) ? replace_ambigs_ : dang_ambigs_;
      for (i = 0; i < print_table.size(); ++i) {
        AmbigSpec_LIST *lst = print_table[i];
        if (lst == nullptr) {
          continue;
        }
        if (!lst->empty()) {
          tprintf("%s Ambiguities for %s:\n", (tbl == 0) ? "Replaceable" : "Dangerous",
                  unicharset->debug_str(i).c_str());
        }
        AmbigSpec_IT lst_it(lst);
        for (lst_it.mark_cycle_pt(); !lst_it.cycled_list(); lst_it.forward()) {
          AmbigSpec *ambig_spec = lst_it.data();
          tprintf("wrong_ngram:");
          UnicharIdArrayUtils::print(ambig_spec->wrong_ngram, *unicharset);
          tprintf("correct_fragments:");
          UnicharIdArrayUtils::print(ambig_spec->correct_fragments, *unicharset);
        }
      }
    }
    if (use_ambigs_for_adaption) {
      for (int vec_id = 0; vec_id < 2; ++vec_id) {
        const std::vector<UnicharIdVector *> &vec =
            (vec_id == 0) ? ambigs_for_adaption_ : reverse_ambigs_for_adaption_;
        for (i = 0; i < vec.size(); ++i) {
          adaption_ambigs_entry = vec[i];
          if (adaption_ambigs_entry != nullptr) {
            tprintf("%sAmbigs for adaption for %s:\n", (vec_id == 0) ? "" : "Reverse ",
                    unicharset->debug_str(i).c_str());
            for (j = 0; j < adaption_ambigs_entry->size(); ++j) {
              tprintf("%s ", unicharset->debug_str((*adaption_ambigs_entry)[j]).c_str());
            }
            tprintf("\n");
          }
        }
      }
    }
  }
}
