    static bool ipv6 = false;
    void enableIPv6(bool state) { ipv6 = state; }
    bool IPv6Enabled() { return ipv6; }

    string getAddrInfoStrError(int code) { 
#if !defined(_WIN32)
        return gai_strerror(code);
