  curl_easy_cleanup(data->req.doh.probe[1].easy);
  data->req.doh.probe[1].easy = NULL;
  return NULL;
}

