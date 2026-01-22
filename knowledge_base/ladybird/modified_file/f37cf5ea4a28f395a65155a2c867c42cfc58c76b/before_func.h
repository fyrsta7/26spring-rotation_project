        Connection(const StringView& address)
            : m_connection(CLocalSocket::construct(this))
            , m_notifier(CNotifier::construct(m_connection->fd(), CNotifier::Read, this))
        {
            // We want to rate-limit our clients
            m_connection->set_blocking(true);
            m_notifier->on_ready_to_read = [this] {
                drain_messages_from_server();
                CEventLoop::current().post_event(*this, make<PostProcessEvent>(m_connection->fd()));
            };

            int retries = 1000;
            while (retries) {
                if (m_connection->connect(CSocketAddress::local(address))) {
                    break;
                }

                dbgprintf("Client::Connection: connect failed: %d, %s\n", errno, strerror(errno));
                sleep(1);
                --retries;
            }
            ASSERT(m_connection->is_connected());
        }
