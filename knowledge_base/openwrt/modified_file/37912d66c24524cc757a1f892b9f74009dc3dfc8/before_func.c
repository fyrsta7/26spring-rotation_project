
static void fe_pending_work(struct work_struct *work)
{
	struct fe_priv *priv = container_of(work, struct fe_priv, pending_work);
	int i;
	bool pending;

	for (i = 0; i < ARRAY_SIZE(fe_work); i++) {
		pending = test_and_clear_bit(fe_work[i].bitnr,
				priv->pending_flags);
		if (pending)
			fe_work[i].action(priv);
	}
}

static int fe_probe(struct platform_device *pdev)
{
	struct resource *res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	const struct of_device_id *match;
	struct fe_soc_data *soc;
	struct net_device *netdev;
	struct fe_priv *priv;
	struct clk *sysclk;
	int err, napi_weight;

	device_reset(&pdev->dev);

	match = of_match_device(of_fe_match, &pdev->dev);
	soc = (struct fe_soc_data *) match->data;

	if (soc->reg_table)
		fe_reg_table = soc->reg_table;
	else
		soc->reg_table = fe_reg_table;

	fe_base = devm_ioremap_resource(&pdev->dev, res);
	if (!fe_base) {
		err = -EADDRNOTAVAIL;
		goto err_out;
	}

	netdev = alloc_etherdev(sizeof(*priv));
	if (!netdev) {
		dev_err(&pdev->dev, "alloc_etherdev failed\n");
		err = -ENOMEM;
		goto err_iounmap;
	}

	SET_NETDEV_DEV(netdev, &pdev->dev);
	netdev->netdev_ops = &fe_netdev_ops;
	netdev->base_addr = (unsigned long) fe_base;

	netdev->irq = platform_get_irq(pdev, 0);
	if (netdev->irq < 0) {
		dev_err(&pdev->dev, "no IRQ resource found\n");
		err = -ENXIO;
		goto err_free_dev;
	}

	if (soc->init_data)
		soc->init_data(soc, netdev);
	/* fake NETIF_F_HW_VLAN_CTAG_RX for good GRO performance */
	netdev->hw_features |= NETIF_F_HW_VLAN_CTAG_RX;
	netdev->vlan_features = netdev->hw_features &
		~(NETIF_F_HW_VLAN_CTAG_TX | NETIF_F_HW_VLAN_CTAG_RX);
	netdev->features |= netdev->hw_features;

	/* fake rx vlan filter func. to support tx vlan offload func */
	if (fe_reg_table[FE_REG_FE_DMA_VID_BASE])
		netdev->features |= NETIF_F_HW_VLAN_CTAG_FILTER;

	priv = netdev_priv(netdev);
	spin_lock_init(&priv->page_lock);
	if (fe_reg_table[FE_REG_FE_COUNTER_BASE]) {
		priv->hw_stats = kzalloc(sizeof(*priv->hw_stats), GFP_KERNEL);
		if (!priv->hw_stats) {
			err = -ENOMEM;
			goto err_free_dev;
		}
		spin_lock_init(&priv->hw_stats->stats_lock);
	}

	sysclk = devm_clk_get(&pdev->dev, NULL);
	if (!IS_ERR(sysclk))
		priv->sysclk = clk_get_rate(sysclk);

	priv->netdev = netdev;
	priv->device = &pdev->dev;
	priv->soc = soc;
	priv->msg_enable = netif_msg_init(fe_msg_level, FE_DEFAULT_MSG_ENABLE);
	priv->frag_size = fe_max_frag_size(ETH_DATA_LEN);
	priv->rx_buf_size = fe_max_buf_size(priv->frag_size);
	priv->tx_ring_size = priv->rx_ring_size = NUM_DMA_DESC;
	INIT_WORK(&priv->pending_work, fe_pending_work);

	napi_weight = 32;
	if (priv->flags & FE_FLAG_NAPI_WEIGHT) {
		napi_weight *= 2;
		priv->tx_ring_size *= 2;
		priv->rx_ring_size *= 2;
	}
	netif_napi_add(netdev, &priv->rx_napi, fe_poll, napi_weight);
	fe_set_ethtool_ops(netdev);

	err = register_netdev(netdev);
	if (err) {
		dev_err(&pdev->dev, "error bringing up device\n");
		goto err_free_dev;
	}
