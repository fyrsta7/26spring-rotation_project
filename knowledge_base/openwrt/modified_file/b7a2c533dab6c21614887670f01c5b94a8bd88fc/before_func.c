
		while (n > 0) {
			ag71xx_wr(ag, AG71XX_REG_TX_STATUS, TX_STATUS_PS);
			n--;
		}
	}

	DBG("%s: %d packets sent out\n", ag->dev->name, sent);

	ag->dev->stats.tx_bytes += bytes_compl;
	ag->dev->stats.tx_packets += sent;

	if (!sent)
		return 0;

	netdev_completed_queue(ag->dev, sent, bytes_compl);
	if ((ring->curr - ring->dirty) < (ring->size * 3) / 4)
		netif_wake_queue(ag->dev);

	return sent;
}

static int ag71xx_rx_packets(struct ag71xx *ag, int limit)
{
	struct net_device *dev = ag->dev;
	struct ag71xx_ring *ring = &ag->rx_ring;
	int offset = ag71xx_buffer_offset(ag);
	unsigned int pktlen_mask = ag->desc_pktlen_mask;
	int done = 0;

	DBG("%s: rx packets, limit=%d, curr=%u, dirty=%u\n",
			dev->name, limit, ring->curr, ring->dirty);

	while (done < limit) {
		unsigned int i = ring->curr % ring->size;
		struct ag71xx_desc *desc = ag71xx_ring_desc(ring, i);
		struct sk_buff *skb;
		int pktlen;
		int err = 0;

		if (ag71xx_desc_empty(desc))
			break;

		if ((ring->dirty + ring->size) == ring->curr) {
			ag71xx_assert(0);
			break;
		}

		ag71xx_wr(ag, AG71XX_REG_RX_STATUS, RX_STATUS_PR);

		pktlen = desc->ctrl & pktlen_mask;
		pktlen -= ETH_FCS_LEN;

		dma_unmap_single(&dev->dev, ring->buf[i].dma_addr,
				 ag->rx_buf_size, DMA_FROM_DEVICE);

		dev->stats.rx_packets++;
		dev->stats.rx_bytes += pktlen;

		skb = build_skb(ring->buf[i].rx_buf, ag71xx_buffer_size(ag));
		if (!skb) {
			skb_free_frag(ring->buf[i].rx_buf);
			goto next;
		}

		skb_reserve(skb, offset);
		skb_put(skb, pktlen);

		if (ag71xx_has_ar8216(ag))
			err = ag71xx_remove_ar8216_header(ag, skb, pktlen);

		if (err) {
			dev->stats.rx_dropped++;
