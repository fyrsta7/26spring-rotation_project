	if (idx == hwidx) {
		/* read hw index again make sure no new tx packet */
		hwidx = fe_reg_r32(FE_REG_TX_DTX_IDX0);
		if (idx == hwidx)
			fe_reg_w32(tx_intr, FE_REG_FE_INT_STATUS);
		else
			*tx_again = 1;
	} else {
		*tx_again = 1;
	}

	if (done) {
		netdev_completed_queue(netdev, done, bytes_compl);
		smp_mb();
		if (unlikely(netif_queue_stopped(netdev) &&
			     (fe_empty_txd(ring) > ring->tx_thresh)))
			netif_wake_queue(netdev);
	}

	return done;
}

static int fe_poll(struct napi_struct *napi, int budget)
{
	struct fe_priv *priv = container_of(napi, struct fe_priv, rx_napi);
	struct fe_hw_stats *hwstat = priv->hw_stats;
	int tx_done, rx_done, tx_again;
	u32 status, fe_status, status_reg, mask;
	u32 tx_intr, rx_intr, status_intr;

	status = fe_reg_r32(FE_REG_FE_INT_STATUS);
	fe_status = status;
	tx_intr = priv->soc->tx_int;
	rx_intr = priv->soc->rx_int;
	status_intr = priv->soc->status_int;
	tx_done = 0;
	rx_done = 0;
	tx_again = 0;

	if (fe_reg_table[FE_REG_FE_INT_STATUS2]) {
		fe_status = fe_reg_r32(FE_REG_FE_INT_STATUS2);
		status_reg = FE_REG_FE_INT_STATUS2;
	} else {
		status_reg = FE_REG_FE_INT_STATUS;
	}

	if (status & tx_intr)
		tx_done = fe_poll_tx(priv, budget, tx_intr, &tx_again);

	if (status & rx_intr)
		rx_done = fe_poll_rx(napi, budget, priv, rx_intr);

	if (unlikely(fe_status & status_intr)) {
		if (hwstat && spin_trylock(&hwstat->stats_lock)) {
			fe_stats_update(priv);
			spin_unlock(&hwstat->stats_lock);
		}
		fe_reg_w32(status_intr, status_reg);
	}

	if (unlikely(netif_msg_intr(priv))) {
		mask = fe_reg_r32(FE_REG_FE_INT_ENABLE);
