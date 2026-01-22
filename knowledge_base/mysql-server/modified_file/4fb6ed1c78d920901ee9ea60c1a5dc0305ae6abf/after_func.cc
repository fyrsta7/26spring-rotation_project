      sin6 = (struct sockaddr_in6 *)ifap->ifa_addr;
      if (IN6_IS_ADDR_LINKLOCAL(&sin6->sin6_addr) ||
          IN6_IS_ADDR_MC_LINKLOCAL(&sin6->sin6_addr))
        continue;
    }
    addrlen = static_cast<socklen_t>((family == AF_INET)
                                         ? sizeof(struct sockaddr_in)
                                         : sizeof(struct sockaddr_in6));
    ret =
        getnameinfo(ifap->ifa_addr, addrlen, buf,
                    static_cast<socklen_t>(sizeof(buf)), NULL, 0, NI_NAMEREQD);
  }
  if (ret != EAI_NONAME && ret != 0) {
    throw LocalHostnameResolutionError(
        "Could not get local hostname: " + std::string(gai_strerror(ret)) +
        " (ret: " + std::to_string(ret) + ", errno: " + std::to_string(errno) +
        ")");
  }
