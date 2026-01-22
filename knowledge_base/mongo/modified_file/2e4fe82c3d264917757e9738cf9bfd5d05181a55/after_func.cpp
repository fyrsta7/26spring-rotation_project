                size_t target = _roundRobinCounter;
                _roundRobinCounter = (_roundRobinCounter + 1) % _consumers.size();

                if (_consumers[target]->appendDocument(std::move(input), _maxBufferSize))
                    return target;
            } break;
            case ExchangePolicyEnum::kKeyRange: {
                size_t target = getTargetConsumer(input.getDocument());
                bool full = _consumers[target]->appendDocument(std::move(input), _maxBufferSize);
                if (full && _orderPreserving) {
                    // TODO send the high watermark here.
                }
                if (full)
                    return target;
            } break;
            default:
                MONGO_UNREACHABLE;
        }
    }

    invariant(input.isEOF());

    // We have reached the end so send EOS to all consumers.
    for (auto& c : _consumers) {
        [[maybe_unused]] auto full = c->appendDocument(input, _maxBufferSize);
    }

    return kInvalidThreadId;
}

size_t Exchange::getTargetConsumer(const Document& input) {
    // Build the key.
    BSONObjBuilder kb;
    size_t counter = 0;
    for (const auto& elem : _keyPattern) {
        auto value = input.getNestedField(_keyPaths[counter]);

