	};

	register_mtd_user(&not);
}


void __init ath79_register_m25p80_multi(struct flash_platform_data *pdata)
{
