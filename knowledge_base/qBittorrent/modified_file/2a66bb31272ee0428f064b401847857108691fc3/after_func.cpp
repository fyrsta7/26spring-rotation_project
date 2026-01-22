
bool PeerInfo::useI2PSocket() const
{
    return static_cast<bool>(m_nativeInfo.flags & lt::peer_info::i2p_socket);
}

bool PeerInfo::useUTPSocket() const
{
