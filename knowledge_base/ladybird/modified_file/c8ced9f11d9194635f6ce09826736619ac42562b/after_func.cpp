    : IPC::ServerConnection<AudioClientEndpoint, AudioServerEndpoint>(*this, "/tmp/portal/audio")
{
}

void ClientConnection::enqueue(const Buffer& buffer)
{
    for (;;) {
        auto success = enqueue_buffer(buffer.anonymous_buffer(), buffer.id(), buffer.sample_count());
        if (success)
            break;
        // FIXME: We don't know what is a good value for this.
