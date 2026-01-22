    evaluate_block_conditions();
    return KSuccess;
}

KResult Socket::setsockopt(int level, int option, Userspace<const void*> user_value, socklen_t user_value_size)
{
    if (level != SOL_SOCKET)
        return ENOPROTOOPT;
    VERIFY(level == SOL_SOCKET);
    switch (option) {
    case SO_SNDTIMEO:
        if (user_value_size != sizeof(timeval))
            return EINVAL;
        m_send_timeout = TRY(copy_time_from_user(static_ptr_cast<timeval const*>(user_value)));
        return KSuccess;
    case SO_RCVTIMEO:
        if (user_value_size != sizeof(timeval))
            return EINVAL;
        m_receive_timeout = TRY(copy_time_from_user(static_ptr_cast<timeval const*>(user_value)));
        return KSuccess;
    case SO_BINDTODEVICE: {
        if (user_value_size != IFNAMSIZ)
            return EINVAL;
        auto user_string = static_ptr_cast<const char*>(user_value);
        auto ifname = TRY(try_copy_kstring_from_user(user_string, user_value_size));
        auto device = NetworkingManagement::the().lookup_by_name(ifname->view());
        if (!device)
            return ENODEV;
        m_bound_interface = move(device);
        return KSuccess;
    }
    case SO_KEEPALIVE:
        // FIXME: Obviously, this is not a real keepalive.
        return KSuccess;
    case SO_TIMESTAMP:
        if (user_value_size != sizeof(int))
            return EINVAL;
        {
            int timestamp;
            TRY(copy_from_user(&timestamp, static_ptr_cast<const int*>(user_value)));
            m_timestamp = timestamp;
        }
        if (m_timestamp && (domain() != AF_INET || type() == SOCK_STREAM)) {
            // FIXME: Support SO_TIMESTAMP for more protocols?
            m_timestamp = 0;
            return ENOTSUP;
        }
        return KSuccess;
    default:
