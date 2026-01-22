void RemoteDebuggerPeerTCP::_thread_func(void *p_ud) {
	RemoteDebuggerPeerTCP *peer = (RemoteDebuggerPeerTCP *)p_ud;
	while (peer->running && peer->is_peer_connected()) {
		peer->_poll();
		if (!peer->is_peer_connected()) {
			break;
		}
		peer->tcp_client->poll(NetSocket::POLL_TYPE_IN_OUT, 1);
	}
}
