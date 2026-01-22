bool CheckWork(CBlock* pblock, CWallet& wallet, CReserveKey& reservekey)
{
    uint256 hash = pblock->GetHash();
    uint256 hashTarget = CBigNum().SetCompact(pblock->nBits).getuint256();

    if (hash > hashTarget)
        return false;

    //// debug print
    printf("BitcoinMiner:\n");
    printf("proof-of-work found  \n  hash: %s  \ntarget: %s\n", hash.GetHex().c_str(), hashTarget.GetHex().c_str());
    pblock->print();
    printf("%s ", DateTimeStrFormat("%x %H:%M", GetTime()).c_str());
    printf("generated %s\n", FormatMoney(pblock->vtx[0].vout[0].nValue).c_str());

    // Found a solution
    CRITICAL_BLOCK(cs_main)
    {
        if (pblock->hashPrevBlock != hashBestChain)
            return error("BitcoinMiner : generated block is stale");

        // Remove key from key pool
        reservekey.KeepKey();

        // Track how many getdata requests this block gets
        CRITICAL_BLOCK(wallet.cs_wallet)
            wallet.mapRequestCount[pblock->GetHash()] = 0;

        // Process this block the same as if we had received it from another node
        if (!ProcessBlock(NULL, pblock))
            return error("BitcoinMiner : ProcessBlock, block not accepted");
    }

    Sleep(2000);
    return true;
}
