
    return mac_result.release_value();
}

ssize_t TLSv12::handle_message(ReadonlyBytes buffer)
{
    auto res { 5ll };
    size_t header_size = res;
    ssize_t payload_res = 0;

    dbgln_if(TLS_DEBUG, "buffer size: {}", buffer.size());

    if (buffer.size() < 5) {
        return (i8)Error::NeedMoreData;
    }

    auto type = (ContentType)buffer[0];
    size_t buffer_position { 1 };

    // FIXME: Read the version and verify it

    if constexpr (TLS_DEBUG) {
        auto version = static_cast<ProtocolVersion>(ByteReader::load16(buffer.offset_pointer(buffer_position)));
        dbgln("type={}, version={}", enum_to_string(type), enum_to_string(version));
    }

    buffer_position += 2;

    auto length = AK::convert_between_host_and_network_endian(ByteReader::load16(buffer.offset_pointer(buffer_position)));

    dbgln_if(TLS_DEBUG, "record length: {} at offset: {}", length, buffer_position);
    buffer_position += 2;

    if (buffer_position + length > buffer.size()) {
        dbgln_if(TLS_DEBUG, "record length more than what we have: {}", buffer.size());
        return (i8)Error::NeedMoreData;
    }

    dbgln_if(TLS_DEBUG, "message type: {}, length: {}", enum_to_string(type), length);
    auto plain = buffer.slice(buffer_position, buffer.size() - buffer_position);

    ByteBuffer decrypted;

    if (m_context.cipher_spec_set && type != ContentType::CHANGE_CIPHER_SPEC) {
        if constexpr (TLS_DEBUG) {
            dbgln("Encrypted: ");
            print_buffer(buffer.slice(header_size, length));
        }

        Error return_value = Error::NoError;
        m_cipher_remote.visit(
            [&](Empty&) { VERIFY_NOT_REACHED(); },
            [&](Crypto::Cipher::AESCipher::GCMMode& gcm) {
                VERIFY(is_aead());
                if (length < 24) {
                    dbgln("Invalid packet length");
                    auto packet = build_alert(true, (u8)AlertDescription::DECRYPT_ERROR);
                    write_packet(packet);
                    return_value = Error::BrokenPacket;
                    return;
                }

                auto packet_length = length - iv_length() - 16;
                auto payload = plain;
                auto decrypted_result = ByteBuffer::create_uninitialized(packet_length);
                if (decrypted_result.is_error()) {
                    dbgln("Failed to allocate memory for the packet");
                    return_value = Error::DecryptionFailed;
                    return;
                }
                decrypted = decrypted_result.release_value();

                // AEAD AAD (13)
                // Seq. no (8)
                // content type (1)
                // version (2)
                // length (2)
                u8 aad[13];
                Bytes aad_bytes { aad, 13 };
                FixedMemoryStream aad_stream { aad_bytes };

                u64 seq_no = AK::convert_between_host_and_network_endian(m_context.remote_sequence_number);
                u16 len = AK::convert_between_host_and_network_endian((u16)packet_length);

                MUST(aad_stream.write_value(seq_no));                                    // sequence number
                MUST(aad_stream.write_until_depleted(buffer.slice(0, header_size - 2))); // content-type + version
                MUST(aad_stream.write_value(len));                                       // length
                VERIFY(MUST(aad_stream.tell()) == MUST(aad_stream.size()));

                auto nonce = payload.slice(0, iv_length());
                payload = payload.slice(iv_length());

                // AEAD IV (12)
                // IV (4)
                // (Nonce) (8)
                // -- Our GCM impl takes 16 bytes
                // zero (4)
                u8 iv[16];
                Bytes iv_bytes { iv, 16 };
                Bytes { m_context.crypto.remote_aead_iv, 4 }.copy_to(iv_bytes);
                nonce.copy_to(iv_bytes.slice(4));
                memset(iv_bytes.offset(12), 0, 4);

                auto ciphertext = payload.slice(0, payload.size() - 16);
                auto tag = payload.slice(ciphertext.size());

                auto consistency = gcm.decrypt(
                    ciphertext,
                    decrypted,
                    iv_bytes,
                    aad_bytes,
                    tag);

                if (consistency != Crypto::VerificationConsistency::Consistent) {
                    dbgln("integrity check failed (tag length {})", tag.size());
                    auto packet = build_alert(true, (u8)AlertDescription::BAD_RECORD_MAC);
                    write_packet(packet);

                    return_value = Error::IntegrityCheckFailed;
                    return;
                }

                plain = decrypted;
            },
            [&](Crypto::Cipher::AESCipher::CBCMode& cbc) {
                VERIFY(!is_aead());
                auto iv_size = iv_length();

                auto decrypted_result = cbc.create_aligned_buffer(length - iv_size);
                if (decrypted_result.is_error()) {
                    dbgln("Failed to allocate memory for the packet");
                    return_value = Error::DecryptionFailed;
                    return;
                }
                decrypted = decrypted_result.release_value();
                auto iv = buffer.slice(header_size, iv_size);

                Bytes decrypted_span = decrypted;
                cbc.decrypt(buffer.slice(header_size + iv_size, length - iv_size), decrypted_span, iv);

                length = decrypted_span.size();

                if constexpr (TLS_DEBUG) {
                    dbgln("Decrypted: ");
                    print_buffer(decrypted);
                }

                auto mac_size = mac_length();
                if (length < mac_size) {
                    dbgln("broken packet");
                    auto packet = build_alert(true, (u8)AlertDescription::DECRYPT_ERROR);
                    write_packet(packet);
                    return_value = Error::BrokenPacket;
                    return;
                }

                length -= mac_size;

                const u8* message_hmac = decrypted_span.offset(length);
                u8 temp_buf[5];
                memcpy(temp_buf, buffer.offset_pointer(0), 3);
                *(u16*)(temp_buf + 3) = AK::convert_between_host_and_network_endian(length);
                auto hmac = hmac_message({ temp_buf, 5 }, decrypted_span.slice(0, length), mac_size);
                auto message_mac = ReadonlyBytes { message_hmac, mac_size };
                if (hmac != message_mac) {
                    dbgln("integrity check failed (mac length {})", mac_size);
                    dbgln("mac received:");
                    print_buffer(message_mac);
                    dbgln("mac computed:");
                    print_buffer(hmac);
                    auto packet = build_alert(true, (u8)AlertDescription::BAD_RECORD_MAC);
                    write_packet(packet);

                    return_value = Error::IntegrityCheckFailed;
                    return;
                }
                plain = decrypted.bytes().slice(0, length);
            });

        if (return_value != Error::NoError) {
            return (i8)return_value;
        }
    }
    m_context.remote_sequence_number++;

    switch (type) {
    case ContentType::APPLICATION_DATA:
        if (m_context.connection_status != ConnectionStatus::Established) {
            dbgln("unexpected application data");
            payload_res = (i8)Error::UnexpectedMessage;
            auto packet = build_alert(true, (u8)AlertDescription::UNEXPECTED_MESSAGE);
            write_packet(packet);
        } else {
            dbgln_if(TLS_DEBUG, "application data message of size {}", plain.size());

            if (m_context.application_buffer.try_append(plain).is_error()) {
                payload_res = (i8)Error::DecryptionFailed;
                auto packet = build_alert(true, (u8)AlertDescription::DECRYPTION_FAILED_RESERVED);
                write_packet(packet);
            } else {
                notify_client_for_app_data();
            }
        }
        break;
    case ContentType::HANDSHAKE:
        dbgln_if(TLS_DEBUG, "tls handshake message");
        payload_res = handle_handshake_payload(plain);
        break;
    case ContentType::CHANGE_CIPHER_SPEC:
        if (m_context.connection_status != ConnectionStatus::KeyExchange) {
            dbgln("unexpected change cipher message");
            auto packet = build_alert(true, (u8)AlertDescription::UNEXPECTED_MESSAGE);
            write_packet(packet);
            payload_res = (i8)Error::UnexpectedMessage;
        } else {
            dbgln_if(TLS_DEBUG, "change cipher spec message");
            m_context.cipher_spec_set = true;
            m_context.remote_sequence_number = 0;
        }
        break;
    case ContentType::ALERT:
        dbgln_if(TLS_DEBUG, "alert message of length {}", length);
        if (length >= 2) {
            if constexpr (TLS_DEBUG)
                print_buffer(plain);

            auto level = plain[0];
            auto code = plain[1];
            dbgln_if(TLS_DEBUG, "Alert received with level {}, code {}", level, code);

            if (level == (u8)AlertLevel::FATAL) {
                dbgln("We were alerted of a critical error: {} ({})", code, enum_to_string((AlertDescription)code));
                m_context.critical_error = code;
                try_disambiguate_error();
                res = (i8)Error::UnknownError;
            }

            if (code == (u8)AlertDescription::CLOSE_NOTIFY) {
                res += 2;
                alert(AlertLevel::FATAL, AlertDescription::CLOSE_NOTIFY);
                if (!m_context.cipher_spec_set) {
                    // AWS CloudFront hits this.
                    dbgln("Server sent a close notify and we haven't agreed on a cipher suite. Treating it as a handshake failure.");
                    m_context.critical_error = (u8)AlertDescription::HANDSHAKE_FAILURE;
                    try_disambiguate_error();
                }
                m_context.close_notify = true;
            }
            m_context.error_code = (Error)code;
            check_connection_state(false);
            notify_client_for_app_data(); // Give the user one more chance to observe the EOF
        }
        break;
    default:
        dbgln("message not understood");
        return (i8)Error::NotUnderstood;
    }

    if (payload_res < 0)
        return payload_res;

    if (res > 0)
