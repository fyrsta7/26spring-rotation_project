
  int32_t len = encodeUdfResponse(NULL, &rsp);
  rsp.msgLen = len;
  void *bufBegin = taosMemoryMalloc(len);
  void *buf = bufBegin;
  encodeUdfResponse(&buf, &rsp);

  uvUdf->output = uv_buf_init(bufBegin, len);

  taosMemoryFree(uvUdf->input.base);
  return;
}

void udfdProcessCallRequest(SUvUdfWork *uvUdf, SUdfRequest *request) {
  SUdfCallRequest *call = &request->call;
  fnDebug("call request. call type %d, handle: %" PRIx64 ", seq num %" PRId64, call->callType, call->udfHandle,
          request->seqNum);
  SUdfcFuncHandle  *handle = (SUdfcFuncHandle *)(call->udfHandle);
  SUdf             *udf = handle->udf;
  SUdfResponse      response = {0};
  SUdfResponse     *rsp = &response;
  SUdfCallResponse *subRsp = &rsp->callRsp;

  int32_t code = TSDB_CODE_SUCCESS;
  switch (call->callType) {
    case TSDB_UDF_CALL_SCALA_PROC: {
      SUdfColumn output = {0};
      output.colMeta.bytes = udf->outputLen;
      output.colMeta.type = udf->outputType;
      output.colMeta.precision = 0;
      output.colMeta.scale = 0;
      udfColEnsureCapacity(&output, call->block.info.rows);

      SUdfDataBlock input = {0};
      convertDataBlockToUdfDataBlock(&call->block, &input);
      code = udf->scriptPlugin->udfScalarProcFunc(&input, &output, udf->scriptUdfCtx);
      freeUdfDataDataBlock(&input);
      convertUdfColumnToDataBlock(&output, &response.callRsp.resultData);
      freeUdfColumn(&output);
      break;
    }
    case TSDB_UDF_CALL_AGG_INIT: {
      SUdfInterBuf outBuf = {.buf = taosMemoryMalloc(udf->bufSize), .bufLen = udf->bufSize, .numOfResult = 0};
      code = udf->scriptPlugin->udfAggStartFunc(&outBuf, udf->scriptUdfCtx);
      subRsp->resultBuf = outBuf;
      break;
    }
    case TSDB_UDF_CALL_AGG_PROC: {
      SUdfDataBlock input = {0};
      convertDataBlockToUdfDataBlock(&call->block, &input);
      SUdfInterBuf outBuf = {.buf = taosMemoryMalloc(udf->bufSize), .bufLen = udf->bufSize, .numOfResult = 0};
      code = udf->scriptPlugin->udfAggProcFunc(&input, &call->interBuf, &outBuf, udf->scriptUdfCtx);
      freeUdfInterBuf(&call->interBuf);
      freeUdfDataDataBlock(&input);
      subRsp->resultBuf = outBuf;

      break;
    }
    case TSDB_UDF_CALL_AGG_MERGE: {
      SUdfInterBuf outBuf = {.buf = taosMemoryMalloc(udf->bufSize), .bufLen = udf->bufSize, .numOfResult = 0};
      code = udf->scriptPlugin->udfAggMergeFunc(&call->interBuf, &call->interBuf2, &outBuf, udf->scriptUdfCtx);
      freeUdfInterBuf(&call->interBuf);
      freeUdfInterBuf(&call->interBuf2);
      subRsp->resultBuf = outBuf;

      break;
    }
    case TSDB_UDF_CALL_AGG_FIN: {
      SUdfInterBuf outBuf = {.buf = taosMemoryMalloc(udf->bufSize), .bufLen = udf->bufSize, .numOfResult = 0};
      code = udf->scriptPlugin->udfAggFinishFunc(&call->interBuf, &outBuf, udf->scriptUdfCtx);
      freeUdfInterBuf(&call->interBuf);
      subRsp->resultBuf = outBuf;
      break;
    }
    default:
      break;
  }

  rsp->seqNum = request->seqNum;
  rsp->type = request->type;
  rsp->code = (code != 0) ? TSDB_CODE_UDF_FUNC_EXEC_FAILURE : 0;
  subRsp->callType = call->callType;

  int32_t len = encodeUdfResponse(NULL, rsp);
  rsp->msgLen = len;
  void *bufBegin = taosMemoryMalloc(len);
  void *buf = bufBegin;
  encodeUdfResponse(&buf, rsp);
  uvUdf->output = uv_buf_init(bufBegin, len);

  switch (call->callType) {
    case TSDB_UDF_CALL_SCALA_PROC: {
      blockDataFreeRes(&call->block);
      blockDataFreeRes(&subRsp->resultData);
      break;
    }
    case TSDB_UDF_CALL_AGG_INIT: {
      freeUdfInterBuf(&subRsp->resultBuf);
      break;
    }
    case TSDB_UDF_CALL_AGG_PROC: {
      blockDataFreeRes(&call->block);
      freeUdfInterBuf(&subRsp->resultBuf);
      break;
    }
    case TSDB_UDF_CALL_AGG_MERGE: {
      freeUdfInterBuf(&subRsp->resultBuf);
