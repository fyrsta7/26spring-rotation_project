  }
  else
  {
    /**
     * 1) This is when a reorg trigger fired...
     *   but the tuple should *not* move
     *   This should be prevented using the LqhKeyReq::setReorgFlag
     *
     * 2) This also happens during reorg copy, when a row should *not* be moved
     */
    jam();
    Uint32 trigOp = regTcPtr->triggeringOperation;
    Uint32 TclientData = regTcPtr->clientData;
    releaseKeys(regCachePtr);
    releaseAttrinfo(cachePtr, apiConnectptr.p);
    regApiPtr->lqhkeyreqrec--;
    unlinkReadyTcCon(apiConnectptr.p);
    clearCommitAckMarker(regApiPtr, regTcPtr);
    releaseTcCon();
    checkPoolShrinkNeed(DBTC_CONNECT_RECORD_TRANSIENT_POOL_INDEX,
                        tcConnectRecord);

    if (trigOp != RNIL)
    {
      jam();
      //ndbassert(false); // see above
      TcConnectRecordPtr opPtr;
      opPtr.i = trigOp;
      tcConnectRecord.getPtr(opPtr);
      ndbrequire(apiConnectptr.p->m_executing_trigger_ops > 0);
      apiConnectptr.p->m_executing_trigger_ops--;
      trigger_op_finished(signal, apiConnectptr, RNIL, opPtr.p, 0);

      ApiConnectRecordPtr transPtr = apiConnectptr;
      executeTriggers(signal, &transPtr);

      return;
    }
    else
    {
      jam();
      Uint32 Ttckeyrec = regApiPtr->tckeyrec;
      regApiPtr->tcSendArray[Ttckeyrec] = TclientData;
      regApiPtr->tcSendArray[Ttckeyrec + 1] = 0;
      regApiPtr->tckeyrec = Ttckeyrec + 2;
      lqhKeyConf_checkTransactionState(signal, apiConnectptr);
    }
  }
}//Dbtc::attrinfoDihReceivedLab()

void Dbtc::packLqhkeyreq(Signal* signal,
                         BlockReference TBRef,
                         CacheRecordPtr cachePtr,
                         ApiConnectRecordPtr const apiConnectptr)
{
  CacheRecord * const regCachePtr = cachePtr.p;
  UintR Tkeylen = regCachePtr->keylen;

  ndbassert( signal->getNoOfSections() == 0 );

  ApiConnectRecord* const regApiPtr = apiConnectptr.p;
  sendlqhkeyreq(signal, TBRef, regCachePtr, regApiPtr);

