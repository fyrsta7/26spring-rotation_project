                                              std::vector<BSONElement>* fixed,
                                              const BSONElement& arrEntry,
                                              BSONObjSet* keys,
                                              unsigned numNotFound,
                                              const BSONElement& arrObjElt,
                                              const std::set<size_t>& arrIdxs,
                                              bool mayExpandArrayUnembedded,
                                              const std::vector<PositionalPathInfo>& positionalInfo,
                                              MultikeyPaths* multikeyPaths) const {
    // Set up any terminal array values.
    for (const auto idx : arrIdxs) {
        if (*(*fieldNames)[idx] == '\0') {
            (*fixed)[idx] = mayExpandArrayUnembedded ? arrEntry : arrObjElt;
        }
    }

    // Recurse.
    getKeysImplWithArray(*fieldNames,
                         *fixed,
                         arrEntry.type() == Object ? arrEntry.embeddedObject() : BSONObj(),
                         keys,
                         numNotFound,
                         positionalInfo,
                         multikeyPaths);
}

void BtreeKeyGeneratorV1::getKeysImpl(std::vector<const char*> fieldNames,
                                      std::vector<BSONElement> fixed,
                                      const BSONObj& obj,
                                      BSONObjSet* keys,
                                      MultikeyPaths* multikeyPaths) const {
    if (_isIdIndex) {
        // we special case for speed
        BSONElement e = obj["_id"];
        if (e.eoo()) {
            keys->insert(_nullKey);
        } else if (_collator) {
            BSONObjBuilder b;
