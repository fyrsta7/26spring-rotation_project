std::string ConsumeScalarRPCArgument(FuzzedDataProvider& fuzzed_data_provider)
{
    const size_t max_string_length = 4096;
    const size_t max_base58_bytes_length{64};
    std::string r;
    CallOneOf(
        fuzzed_data_provider,
        [&] {
            // string argument
            r = fuzzed_data_provider.ConsumeRandomLengthString(max_string_length);
        },
        [&] {
            // base64 argument
            r = EncodeBase64(fuzzed_data_provider.ConsumeRandomLengthString(max_string_length));
        },
        [&] {
            // hex argument
            r = HexStr(fuzzed_data_provider.ConsumeRandomLengthString(max_string_length));
        },
        [&] {
            // bool argument
            r = fuzzed_data_provider.ConsumeBool() ? "true" : "false";
        },
        [&] {
            // range argument
            r = "[" + ToString(fuzzed_data_provider.ConsumeIntegral<int64_t>()) + "," + ToString(fuzzed_data_provider.ConsumeIntegral<int64_t>()) + "]";
        },
        [&] {
            // integral argument (int64_t)
            r = ToString(fuzzed_data_provider.ConsumeIntegral<int64_t>());
        },
        [&] {
            // integral argument (uint64_t)
            r = ToString(fuzzed_data_provider.ConsumeIntegral<uint64_t>());
        },
        [&] {
            // floating point argument
            r = strprintf("%f", fuzzed_data_provider.ConsumeFloatingPoint<double>());
        },
        [&] {
            // tx destination argument
            r = EncodeDestination(ConsumeTxDestination(fuzzed_data_provider));
        },
        [&] {
            // uint160 argument
            r = ConsumeUInt160(fuzzed_data_provider).ToString();
        },
        [&] {
            // uint256 argument
            r = ConsumeUInt256(fuzzed_data_provider).ToString();
        },
        [&] {
            // base32 argument
            r = EncodeBase32(fuzzed_data_provider.ConsumeRandomLengthString(max_string_length));
        },
        [&] {
            // base58 argument
            r = EncodeBase58(MakeUCharSpan(fuzzed_data_provider.ConsumeRandomLengthString(max_base58_bytes_length)));
        },
        [&] {
            // base58 argument with checksum
            r = EncodeBase58Check(MakeUCharSpan(fuzzed_data_provider.ConsumeRandomLengthString(max_base58_bytes_length)));
        },
        [&] {
            // hex encoded block
            std::optional<CBlock> opt_block = ConsumeDeserializable<CBlock>(fuzzed_data_provider);
            if (!opt_block) {
                return;
            }
            CDataStream data_stream{SER_NETWORK, PROTOCOL_VERSION};
            data_stream << *opt_block;
            r = HexStr(data_stream);
        },
        [&] {
            // hex encoded block header
            std::optional<CBlockHeader> opt_block_header = ConsumeDeserializable<CBlockHeader>(fuzzed_data_provider);
            if (!opt_block_header) {
                return;
            }
            CDataStream data_stream{SER_NETWORK, PROTOCOL_VERSION};
            data_stream << *opt_block_header;
            r = HexStr(data_stream);
        },
        [&] {
            // hex encoded tx
            std::optional<CMutableTransaction> opt_tx = ConsumeDeserializable<CMutableTransaction>(fuzzed_data_provider);
            if (!opt_tx) {
                return;
            }
            CDataStream data_stream{SER_NETWORK, fuzzed_data_provider.ConsumeBool() ? PROTOCOL_VERSION : (PROTOCOL_VERSION | SERIALIZE_TRANSACTION_NO_WITNESS)};
            data_stream << *opt_tx;
            r = HexStr(data_stream);
        },
        [&] {
            // base64 encoded psbt
            std::optional<PartiallySignedTransaction> opt_psbt = ConsumeDeserializable<PartiallySignedTransaction>(fuzzed_data_provider);
            if (!opt_psbt) {
                return;
            }
            CDataStream data_stream{SER_NETWORK, PROTOCOL_VERSION};
            data_stream << *opt_psbt;
            r = EncodeBase64({data_stream.begin(), data_stream.end()});
        },
        [&] {
            // base58 encoded key
            const std::vector<uint8_t> random_bytes = fuzzed_data_provider.ConsumeBytes<uint8_t>(32);
            CKey key;
            key.Set(random_bytes.begin(), random_bytes.end(), fuzzed_data_provider.ConsumeBool());
            if (!key.IsValid()) {
                return;
            }
            r = EncodeSecret(key);
        },
        [&] {
            // hex encoded pubkey
            const std::vector<uint8_t> random_bytes = fuzzed_data_provider.ConsumeBytes<uint8_t>(32);
            CKey key;
            key.Set(random_bytes.begin(), random_bytes.end(), fuzzed_data_provider.ConsumeBool());
            if (!key.IsValid()) {
                return;
            }
            r = HexStr(key.GetPubKey());
        });
    return r;
}
