			}
		} else {
			break;
		}
	}
	tx_ring->free_index = index;
	tx_ring->num_used -= i;
	eth_check_num_used(tx_ring);
}

static int eth_poll(struct napi_struct *napi, int budget)
{
	struct sw *sw = container_of(napi, struct sw, napi);
	struct _rx_ring *rx_ring = &sw->rx_ring;
	int received = 0;
	unsigned int length;
	unsigned int i = rx_ring->cur_index;
	struct rx_desc *desc = &(rx_ring)->desc[i];
	unsigned int alloc_count = rx_ring->alloc_count;

	while (desc->cown && alloc_count + received < RX_DESCS - 1) {
		struct sk_buff *skb;
		int reserve = SKB_HEAD_ALIGN;

		if (received >= budget)
			break;

		/* process received frame */
		dma_unmap_single(NULL, rx_ring->phys_tab[i],
				 RX_SEGMENT_MRU, DMA_FROM_DEVICE);

		skb = build_skb(rx_ring->buff_tab[i], 0);
		if (!skb)
			break;

		skb->dev = switch_port_tab[desc->sp]->netdev;

		length = desc->sdl;
		if (desc->fsd && !desc->lsd)
			length = RX_SEGMENT_MRU;

		if (!desc->fsd) {
			reserve -= NET_IP_ALIGN;
			if (!desc->lsd)
				length += NET_IP_ALIGN;
		}

		skb_reserve(skb, reserve);
		skb_put(skb, length);

		if (!sw->frag_first)
			sw->frag_first = skb;
		else {
			if (sw->frag_first == sw->frag_last)
				skb_frag_add_head(sw->frag_first, skb);
			else
				sw->frag_last->next = skb;
			sw->frag_first->len += skb->len;
			sw->frag_first->data_len += skb->len;
			sw->frag_first->truesize += skb->truesize;
		}
		sw->frag_last = skb;

		if (desc->lsd) {
			struct net_device *dev;

			skb = sw->frag_first;
			dev = skb->dev;
			skb->protocol = eth_type_trans(skb, dev);

			dev->stats.rx_packets++;
			dev->stats.rx_bytes += skb->len;

			/* RX Hardware checksum offload */
			skb->ip_summed = CHECKSUM_NONE;
			switch (desc->prot) {
				case 1:
				case 2:
				case 5:
				case 6:
				case 13:
				case 14:
					if (!desc->l4f) {
						skb->ip_summed = CHECKSUM_UNNECESSARY;
						napi_gro_receive(napi, skb);
						break;
					}
					/* fall through */
				default:
					netif_receive_skb(skb);
					break;
			}

			sw->frag_first = NULL;
			sw->frag_last = NULL;
		}

		received++;
		if (++i == RX_DESCS) {
			i = 0;
			desc = &(rx_ring)->desc[i];
		} else {
			desc++;
		}
	}

	if (!received) {
		napi_complete(napi);
		enable_irq(IRQ_CNS3XXX_SW_R0RXC);
	}

	spin_lock_bh(&tx_lock);
	eth_complete_tx(sw);
	spin_unlock_bh(&tx_lock);
