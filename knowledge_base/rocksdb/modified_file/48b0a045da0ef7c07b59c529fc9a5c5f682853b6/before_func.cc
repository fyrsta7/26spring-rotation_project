        fprintf(stdout, "%s ", ReadableTime(rawtime).c_str());
      }
      string str = PrintKeyValue(iter->key().ToString(),
                                 iter->value().ToString(), is_key_hex_,
                                 is_value_hex_);
      fprintf(stdout, "%s\n", str.c_str());
    }
  }

  if (num_buckets > 1 && is_db_ttl_) {
    PrintBucketCounts(bucket_counts, ttl_start, ttl_end, bucket_size,
                      num_buckets);
  } else if(count_delim_) {
    fprintf(stdout,"%s => count:%lld\tsize:%lld\n",rtype2.c_str(),
        (long long )c,(long long)s2);
  } else {
    fprintf(stdout, "Keys in range: %lld\n", (long long) count);
