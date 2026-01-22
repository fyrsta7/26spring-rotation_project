void RemoteDebuggerPeerTCP::_thread_func(void *p_ud) {
	const uint64_t min_tick = 100;
	RemoteDebuggerPeerTCP *peer = (RemoteDebuggerPeerTCP *)p_ud;
	while (peer->running && peer->is_peer_connected()) {
		uint64_t ticks_usec = OS::get_singleton()->get_ticks_usec();
		peer->_poll();
		if (!peer->is_peer_connected()) {
			break;
		}
		ticks_usec = OS::get_singleton()->get_ticks_usec() - ticks_usec;
		if (ticks_usec < min_tick) {
			OS::get_singleton()->delay_usec(min_tick - ticks_usec);
		}
	}
}
