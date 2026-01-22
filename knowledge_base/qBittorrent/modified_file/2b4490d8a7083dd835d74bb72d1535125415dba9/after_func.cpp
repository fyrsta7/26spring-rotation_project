bool InfoHash::isValid() const
{
    return m_valid;
}

InfoHash::operator lt::sha1_hash() const
{
    return m_nativeHash;
