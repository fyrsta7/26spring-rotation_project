      nread = fread(buffer, 1, sizeof(buffer), file);
      if(ferror(file)) {
        curlx_dyn_free(&dyn);
        *size = 0;
        *bufp = NULL;
        return PARAM_READ_ERROR;
      }
      if(nread)
        if(curlx_dyn_addn(&dyn, buffer, nread))
          return PARAM_NO_MEM;
    } while(!feof(file));
    *size = curlx_dyn_len(&dyn);
    *bufp = curlx_dyn_ptr(&dyn);
  }
  else {
    *size = 0;
