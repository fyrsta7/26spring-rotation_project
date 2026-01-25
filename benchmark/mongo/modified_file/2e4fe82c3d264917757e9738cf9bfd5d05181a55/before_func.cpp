size_t Exchange::getTargetConsumer(const Document& input) {
    // Build the key.
    BSONObjBuilder kb;
    size_t counter = 0;
    for (auto elem : _keyPattern) {
        auto value = input.getNestedField(_keyPaths[counter]);

        // By definition we send documents with missing fields to the consumer 0.
        if (value.missing()) {
            return 0;
        }

        if (elem.type() == BSONType::String && elem.str() == "hashed") {
            kb << ""
               << BSONElementHasher::hash64(BSON("" << value).firstElement(),
                                            BSONElementHasher::DEFAULT_HASH_SEED);
        } else {
            kb << "" << value;
        }
        ++counter;
    }

    KeyString::Builder key{KeyString::Version::V1, kb.obj(), _ordering};
    std::string keyStr{key.getBuffer(), key.getSize()};

    // Binary search for the consumer id.
    auto it = std::upper_bound(_boundaries.begin(), _boundaries.end(), keyStr);
    invariant(it != _boundaries.end());

    size_t distance = std::distance(_boundaries.begin(), it) - 1;
    invariant(distance < _consumerIds.size());

    size_t cid = _consumerIds[distance];
    invariant(cid < _consumers.size());

    return cid;
}