 */
static void bcm6368_enetsw_refill_rx_timer(struct timer_list *t)
{
	struct bcm6368_enetsw *priv = from_timer(priv, t, rx_timeout);
	struct net_device *dev = priv->net_dev;

	spin_lock(&priv->rx_lock);
	bcm6368_enetsw_refill_rx(dev, false);
	spin_unlock(&priv->rx_lock);
}

/*
 * extract packet from rx queue
 */
static int bcm6368_enetsw_receive_queue(struct net_device *dev, int budget)
{
	struct bcm6368_enetsw *priv = netdev_priv(dev);
	struct device *kdev = &priv->pdev->dev;
	struct list_head rx_list;
	int processed = 0;

	INIT_LIST_HEAD(&rx_list);

	/* don't scan ring further than number of refilled
	 * descriptor */
	if (budget > priv->rx_desc_count)
		budget = priv->rx_desc_count;

	do {
		struct bcm6368_enetsw_desc *desc;
		unsigned int frag_size;
		struct sk_buff *skb;
		unsigned char *buf;
		int desc_idx;
		u32 len_stat;
		unsigned int len;

		desc_idx = priv->rx_curr_desc;
		desc = &priv->rx_desc_cpu[desc_idx];

		/* make sure we actually read the descriptor status at
		 * each loop */
		rmb();

		len_stat = desc->len_stat;

		/* break if dma ownership belongs to hw */
		if (len_stat & DMADESC_OWNER_MASK)
			break;

		processed++;
		priv->rx_curr_desc++;
		if (priv->rx_curr_desc == priv->rx_ring_size)
			priv->rx_curr_desc = 0;

		/* if the packet does not have start of packet _and_
		 * end of packet flag set, then just recycle it */
		if ((len_stat & DMADESC_ESOP_MASK) != DMADESC_ESOP_MASK) {
			dev->stats.rx_dropped++;
			continue;
		}

		/* valid packet */
		buf = priv->rx_buf[desc_idx];
		len = (len_stat & DMADESC_LENGTH_MASK)
		      >> DMADESC_LENGTH_SHIFT;
		/* don't include FCS */
		len -= 4;

		if (len < priv->copybreak) {
			unsigned int nfrag_size = ENETSW_FRAG_SIZE(len);
			unsigned char *nbuf = napi_alloc_frag(nfrag_size);

			if (unlikely(!nbuf)) {
				/* forget packet, just rearm desc */
				dev->stats.rx_dropped++;
				continue;
			}

			dma_sync_single_for_cpu(kdev, desc->address,
						len, DMA_FROM_DEVICE);
			memcpy(nbuf + NET_SKB_PAD, buf + NET_SKB_PAD, len);
			dma_sync_single_for_device(kdev, desc->address,
						   len, DMA_FROM_DEVICE);
			buf = nbuf;
			frag_size = nfrag_size;
		} else {
			dma_unmap_single(kdev, desc->address,
					 priv->rx_buf_size, DMA_FROM_DEVICE);
			priv->rx_buf[desc_idx] = NULL;
			frag_size = priv->rx_frag_size;
		}

		skb = napi_build_skb(buf, frag_size);
		if (unlikely(!skb)) {
			skb_free_frag(buf);
			dev->stats.rx_dropped++;
			continue;
		}

		skb_reserve(skb, NET_SKB_PAD);
		skb_put(skb, len);
		skb->protocol = eth_type_trans(skb, dev);
		dev->stats.rx_packets++;
