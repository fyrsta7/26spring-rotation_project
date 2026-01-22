				sw_w32_mask(0, BIT(2), priv->r->dma_if_ctrl);
			else
				sw_w32_mask(0, BIT(3), priv->r->dma_if_ctrl);
		} else {
			sw_w32_mask(0, TX_DO, priv->r->dma_if_ctrl);
		}

		dev->stats.tx_packets++;
		dev->stats.tx_bytes += len;
		dev_kfree_skb(skb);
		ring->c_tx[q] = (ring->c_tx[q] + 1) % TXRINGLEN;
		ret = NETDEV_TX_OK;
	} else {
		dev_warn(&priv->pdev->dev, "Data is owned by switch\n");
		ret = NETDEV_TX_BUSY;
	}
txdone:
	spin_unlock_irqrestore(&priv->lock, flags);
	return ret;
}

/*
 * Return queue number for TX. On the RTL83XX, these queues have equal priority
 * so we do round-robin
 */
u16 rtl83xx_pick_tx_queue(struct net_device *dev, struct sk_buff *skb,
			  struct net_device *sb_dev)
{
	static u8 last = 0;

	last++;
	return last % TXRINGS;
}

/*
 * Return queue number for TX. On the RTL93XX, queue 1 is the high priority queue
 */
u16 rtl93xx_pick_tx_queue(struct net_device *dev, struct sk_buff *skb,
			  struct net_device *sb_dev)
{
	if (skb->priority >= TC_PRIO_CONTROL)
		return 1;
	return 0;
}

static int rtl838x_hw_receive(struct net_device *dev, int r, int budget)
{
	struct rtl838x_eth_priv *priv = netdev_priv(dev);
	struct ring_b *ring = priv->membase;
	struct sk_buff *skb;
	unsigned long flags;
	int i, len, work_done = 0;
	u8 *data, *skb_data;
	unsigned int val;
	u32	*last;
	struct p_hdr *h;
	bool dsa = netdev_uses_dsa(dev);
	struct dsa_tag tag;

	pr_debug("---------------------------------------------------------- RX - %d\n", r);
	spin_lock_irqsave(&priv->lock, flags);
	last = (u32 *)KSEG1ADDR(sw_r32(priv->r->dma_if_rx_cur + r * 4));

	do {
		if ((ring->rx_r[r][ring->c_rx[r]] & 0x1)) {
			if (&ring->rx_r[r][ring->c_rx[r]] != last) {
				netdev_warn(dev, "Ring contention: r: %x, last %x, cur %x\n",
				    r, (uint32_t)last, (u32) &ring->rx_r[r][ring->c_rx[r]]);
			}
			break;
		}

		h = &ring->rx_header[r][ring->c_rx[r]];
		data = (u8 *)KSEG1ADDR(h->buf);
		len = h->len;
		if (!len)
			break;
		work_done++;

		len -= 4; /* strip the CRC */
		/* Add 4 bytes for cpu_tag */
		if (dsa)
			len += 4;

		skb = netdev_alloc_skb(dev, len + 4);
		skb_reserve(skb, NET_IP_ALIGN);

		if (likely(skb)) {
			/* BUG: Prevent bug on RTL838x SoCs*/
			if (priv->family_id == RTL8380_FAMILY_ID) {
				sw_w32(0xffffffff, priv->r->dma_if_rx_ring_size(0));
				for (i = 0; i < priv->rxrings; i++) {
					/* Update each ring cnt */
					val = sw_r32(priv->r->dma_if_rx_ring_cntr(i));
					sw_w32(val, priv->r->dma_if_rx_ring_cntr(i));
				}
			}

			skb_data = skb_put(skb, len);
			/* Make sure data is visible */
			mb();
			memcpy(skb->data, (u8 *)KSEG1ADDR(data), len);
			/* Overwrite CRC with cpu_tag */
			if (dsa) {
				priv->r->decode_tag(h, &tag);
				skb->data[len-4] = 0x80;
				skb->data[len-3] = tag.port;
