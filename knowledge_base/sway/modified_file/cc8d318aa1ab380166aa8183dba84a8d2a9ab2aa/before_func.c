}

static void transaction_progress_queue(void) {
	if (!server.transactions->length) {
		return;
	}
	// Only the first transaction in the queue is committed, so that's the one
	// we try to process.
	struct sway_transaction *transaction = server.transactions->items[0];
	if (transaction->num_waiting) {
		return;
	}
	transaction_apply(transaction);
	transaction_destroy(transaction);
	list_del(server.transactions, 0);

	if (server.transactions->length == 0) {
		// The transaction queue is empty, so we're done.
		sway_idle_inhibit_v1_check_active(server.idle_inhibit_manager_v1);
		return;
	}

	// If there's a bunch of consecutive transactions which all apply to the
	// same views, skip all except the last one.
	while (server.transactions->length >= 2) {
		struct sway_transaction *a = server.transactions->items[0];
		struct sway_transaction *b = server.transactions->items[1];
		if (transaction_same_nodes(a, b)) {
			list_del(server.transactions, 0);
			transaction_destroy(a);
		} else {
			break;
		}
	}

	// We again commit the first transaction in the queue to process it.
	transaction = server.transactions->items[0];
	transaction_commit(transaction);
