void WalletModel::pollBalanceChanged()
{
    // Try to get balances and return early if locks can't be acquired. This
    // avoids the GUI from getting stuck on periodical polls if the core is
    // holding the locks for a longer time - for example, during a wallet
    // rescan.
    interfaces::WalletBalances new_balances;
    int numBlocks = -1;
    if (!m_wallet->tryGetBalances(new_balances, numBlocks, fForceCheckBalanceChanged, cachedNumBlocks)) {
        return;
    }

    fForceCheckBalanceChanged = false;

    // Balance and number of transactions might have changed
    cachedNumBlocks = numBlocks;

    checkBalanceChanged(new_balances);
    if(transactionTableModel)
        transactionTableModel->updateConfirmations();
}
