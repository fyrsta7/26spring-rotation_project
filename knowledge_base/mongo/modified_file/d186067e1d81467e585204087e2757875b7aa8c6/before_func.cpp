    while (indexIt->more()) {
        const IndexCatalogEntry* indexEntry = indexIt->next();
        const std::string indexName = indexEntry->descriptor()->indexName();

        Status status =
            index_key_validate::validateIndexSpecFieldNames(indexEntry->descriptor()->infoObj());
        if (!status.isOK()) {
            results->valid = false;
            results->errors.push_back(
                fmt::format("The index specification for index '{}' contains invalid field names. "
                            "{}. Run the 'collMod' command on the collection without any arguments "
                            "to remove the invalid index options",
                            indexName,
                            status.reason()));
        }

        if (!indexEntry->isReady(opCtx, collection)) {
            continue;
        }

        MultikeyPaths multikeyPaths;
        const bool isMultikey = collection->isIndexMultikey(opCtx, indexName, &multikeyPaths);
        const bool hasMultiKeyPaths = std::any_of(multikeyPaths.begin(),
                                                  multikeyPaths.end(),
                                                  [](auto& pathSet) { return pathSet.size() > 0; });
        // It is illegal for multikey paths to exist without the multikey flag set on the index,
        // but it may be possible for multikey to be set on the index while having no multikey
        // paths. If any of the paths are multikey, then the entire index should also be marked
        // multikey.
        if (hasMultiKeyPaths && !isMultikey) {
            results->valid = false;
            results->errors.push_back(
                fmt::format("The 'multikey' field for index {} was false with non-empty "
                            "'multikeyPaths': {}",
                            indexName,
                            multikeyPathsToString(multikeyPaths)));
        }
    }
}

void _validateBSONColumnRoundtrip(OperationContext* opCtx,
                                  ValidateState* validateState,
                                  ValidateResults* results) {
    LOGV2(6104700,
          "Validating BSONColumn compression/decompression",
          "namespace"_attr = validateState->nss());
    std::deque<BSONObj> original;
    auto cursor = validateState->getCollection()->getRecordStore()->getCursor(opCtx);

    // This function is memory intensive as it needs to store the original documents prior to
    // compressing and decompressing them to check that the documents are the same afterwards. We'll
    // limit the number of original documents we hold in-memory to be approximately 100MB to avoid
    // running out of memory.
    constexpr size_t kMaxMemoryUsageBytes = 100 * 1024 * 1024;
    size_t currentMemoryUsageBytes = 0;

    BSONColumnBuilder columnBuilder("");

    auto doBSONColumnRoundtrip = [&]() {
        ON_BLOCK_EXIT([&] {
            // Reset the in-memory state to prepare for the next round of BSONColumn roundtripping.
            original.clear();
            columnBuilder = BSONColumnBuilder("");
            currentMemoryUsageBytes = 0;
        });

        BSONObjBuilder compressed;
        try {
            compressed.append(""_sd, columnBuilder.finalize());

            BSONColumn column(compressed.done().firstElement());
            size_t index = 0;
            for (const auto& decompressed : column) {
                if (!decompressed.binaryEqual(original[index].firstElement())) {
                    results->valid = false;
                    results->errors.push_back(
                        fmt::format("Roundtripping via BSONColumn failed. Index: {}, Original: {}, "
                                    "Roundtripped: {}",
                                    index,
                                    original[index].toString(),
                                    decompressed.toString()));
                    return;
                }
                ++index;
            }
            if (index != original.size()) {
                results->valid = false;
                results->errors.push_back(fmt::format(
                    "Roundtripping via BSONColumn failed. Original size: {}, Roundtripped size: {}",
                    original.size(),
                    index));
            }
        } catch (const DBException&) {
            // We swallow any other DBException so we do not interfere with the rest of Collection
            // validation.
            return;
        }
    };

    while (auto record = cursor->next()) {
        try {
