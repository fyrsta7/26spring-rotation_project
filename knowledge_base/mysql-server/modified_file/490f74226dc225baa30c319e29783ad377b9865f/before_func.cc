                         user, acl_user->user ? acl_user->user : "",
                         host,
                         acl_user->host.get_host() ? acl_user->host.get_host() :
                         ""));
      if ((!acl_user->user && !user[0]) ||
          (acl_user->user && !strcmp(user,acl_user->user)))
      {
        if (exact ? !my_strcasecmp(system_charset_info, host,
                                   acl_user->host.get_host() ?
                                   acl_user->host.get_host() : "") :
            acl_user->host.compare_hostname(host,host))
        {
          DBUG_RETURN(acl_user);
        }
      }
    }
  }
  DBUG_RETURN(0);
