}

void LogSparseFeatureDataLoss(StringPiece feature_name) {
  LOG(WARNING) << "Data loss! Feature '" << feature_name
               << "' is present in multiple concatenated "
                  "tf.Examples. Ignoring all but last one.";
  static auto* duplicated_sparse_feature = monitoring::Counter<0>::New(
      "/tensorflow/core/util/example_proto_fast_parsing/"
      "duplicated_sparse_feature",
      "Sparse feature appears twice in a tf.Example");
  duplicated_sparse_feature->GetCell()->IncrementBy(1);
}

Status FastParseSerializedExample(
    const string& serialized_example, const string& example_name,
    const size_t example_index, const Config& config,
    const PresizedCuckooMap<std::pair<size_t, Type>>& config_index,
    SeededHasher hasher, std::vector<Tensor>* output_dense,
    std::vector<SparseBuffer>* output_varlen_dense,
    std::vector<SparseBuffer>* output_sparse,
    std::vector<SparseBuffer>* output_ragged,
    PerExampleFeatureStats* output_stats) {
  DCHECK(output_dense != nullptr);
  DCHECK(output_sparse != nullptr);
  DCHECK(output_ragged != nullptr);
  parsed::Example parsed_example;
  if (!ParseExample(serialized_example, &parsed_example)) {
    return errors::InvalidArgument("Could not parse example input, value: '",
                                   serialized_example, "'");
  }
  std::vector<int64> sparse_feature_last_example(config.sparse.size(), -1);
  std::vector<int64> dense_feature_last_example(config.dense.size(), -1);
  std::vector<int64> ragged_feature_last_example(config.ragged.size(), -1);

  // Handle features present in the example.
  const size_t parsed_example_size = parsed_example.size();

  if (output_stats) {
    // TODO(b/111553342): This may over-count the number of features if there
    // are duplicate keys in the feature map. Consider deduplicating the keys
    // before computing the count.
    output_stats->features_count = parsed_example_size;
  }

  for (size_t i = 0; i < parsed_example_size; ++i) {
    // This is a logic that standard protobuf parsing is implementing.
    // I.e. last entry in the map overwrites all the previous ones.
    parsed::FeatureMapEntry& name_and_feature =
        parsed_example[parsed_example_size - i - 1];

    const StringPiece feature_name = name_and_feature.first;
    parsed::Feature& feature = name_and_feature.second;

    std::pair<size_t, Type> d_and_type;
    uint64 h = hasher(feature_name);
    if (!config_index.Find(h, &d_and_type)) continue;

    size_t d = d_and_type.first;
    bool is_dense = d_and_type.second == Type::Dense;
    bool is_ragged = d_and_type.second == Type::Ragged;

    {
      // Testing for PresizedCuckooMap collision.
      // TODO(lew): Use dense_hash_map and avoid this and hasher creation.
      const string& config_feature_name =
          is_dense ? config.dense[d].feature_name
                   : (is_ragged ? config.ragged[d].feature_name
                                : config.sparse[d].feature_name);
      if (feature_name != config_feature_name) continue;
    }

    auto example_error = [&](StringPiece suffix) {
      return errors::InvalidArgument("Name: ", example_name,
                                     ", Key: ", feature_name,
                                     ", Index: ", example_index, ".  ", suffix);
    };

    auto parse_error = [&] {
      return example_error("Can't parse serialized Example.");
    };

    DataType example_dtype;
    TF_RETURN_IF_ERROR(feature.ParseDataType(&example_dtype));

    if (is_dense) {
      if (example_dtype == DT_INVALID) continue;

      // If feature was already visited, skip.
      // Compare comment at the beginning of the loop.
      if (dense_feature_last_example[d] == example_index) {
        LogDenseFeatureDataLoss(feature_name);
        continue;
      }
      dense_feature_last_example[d] = example_index;

      if (example_dtype != config.dense[d].dtype) {
        return example_error(strings::StrCat(
            "Data types don't match. Data type: ",
            DataTypeString(example_dtype),
            " but expected type: ", DataTypeString(config.dense[d].dtype)));
      }
      if (!config.dense[d].variable_length) {
        Tensor& out = (*output_dense)[d];

        const std::size_t num_elements = config.dense[d].elements_per_stride;
        if (output_stats) {
          // TODO(b/111553342): If desirable, we could add support for counting
          // elements in the features that aren't parsed, but this could add
          // considerable runtime cost.
          output_stats->feature_values_count += num_elements;
        }

        const std::size_t offset = example_index * num_elements;

        auto shape_error = [&](size_t size, StringPiece type_str) {
          return example_error(strings::StrCat(
              "Number of ", type_str,
              " values != expected.  "
              "Values size: ",
              size,
              " but output shape: ", config.dense[d].shape.DebugString()));
        };

        switch (config.dense[d].dtype) {
          case DT_INT64: {
            auto out_p = out.flat<int64>().data() + offset;
            LimitedArraySlice<int64> slice(out_p, num_elements);
            if (!feature.ParseInt64List(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "int64");
            }
            break;
          }
          case DT_FLOAT: {
            auto out_p = out.flat<float>().data() + offset;
            LimitedArraySlice<float> slice(out_p, num_elements);
            if (!feature.ParseFloatList(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "float");
            }
            break;
          }
          case DT_STRING: {
            auto out_p = out.flat<tstring>().data() + offset;
            LimitedArraySlice<tstring> slice(out_p, num_elements);
            if (!feature.ParseBytesList(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "bytes");
            }
            break;
          }
          default:
            LOG(FATAL) << "Should not happen.";
        }
      } else {  // if variable length
        SparseBuffer& out = (*output_varlen_dense)[d];

        const std::size_t num_elements = config.dense[d].elements_per_stride;

        if (example_dtype != DT_INVALID &&
            example_dtype != config.dense[d].dtype) {
          return example_error(strings::StrCat(
              "Data types don't match. ",
              "Expected type: ", DataTypeString(config.dense[d].dtype)));
        }

        auto shape_error = [&](size_t size, StringPiece type_str) {
          return example_error(strings::StrCat(
              "Number of ", type_str,
              " values is not a multiple of stride length. Saw ", size,
              " values but output shape is: ",
              config.dense[d].shape.DebugString()));
        };

        switch (config.dense[d].dtype) {
          case DT_INT64: {
            if (example_dtype != DT_INVALID) {
              if (!feature.ParseInt64List(&out.int64_list)) {
                return parse_error();
              }
              if (out.int64_list.size() % num_elements != 0) {
                return shape_error(out.int64_list.size(), "int64");
              }
            }
            out.example_end_indices.push_back(out.int64_list.size());
            break;
          }
          case DT_FLOAT: {
            if (example_dtype != DT_INVALID) {
              if (!feature.ParseFloatList(&out.float_list)) {
                return parse_error();
              }
              if (out.float_list.size() % num_elements != 0) {
                return shape_error(out.float_list.size(), "float");
              }
            }
            out.example_end_indices.push_back(out.float_list.size());
            break;
          }
          case DT_STRING: {
            if (example_dtype != DT_INVALID) {
              if (!feature.ParseBytesList(&out.bytes_list)) {
                return parse_error();
              }
              if (out.bytes_list.size() % num_elements != 0) {
                return shape_error(out.bytes_list.size(), "bytes");
              }
            }
            out.example_end_indices.push_back(out.bytes_list.size());
            break;
          }
          default:
            LOG(FATAL) << "Should not happen.";
        }

        if (output_stats) {
          // Use `out.example_end_indices` to determine the feature-value count
          // for this feature, because the preceding switch statement pushes
          // the length of the appropriate feature list to that vector.
          // TODO(b/111553342): If desirable, we could add support for counting
          // elements in the features that aren't parsed, but this could add
          // considerable runtime cost.
          const size_t out_examples_count = out.example_end_indices.size();
          if (out_examples_count == 1) {
            output_stats->feature_values_count += out.example_end_indices[0];
          } else {
            output_stats->feature_values_count +=
                out.example_end_indices[out_examples_count - 1] -
                out.example_end_indices[out_examples_count - 2];
          }
        }
      }
    } else {
      // Feature is sparse or ragged.
      auto& last_example =
          is_ragged ? ragged_feature_last_example : sparse_feature_last_example;

      // If feature was already visited, skip.
      // Compare comment at the beginning of the loop.
      if (last_example[d] == example_index) {
        LogSparseFeatureDataLoss(feature_name);
        continue;
      }
      last_example[d] = example_index;

      // Handle sparse features.
      SparseBuffer& out = is_ragged ? (*output_ragged)[d] : (*output_sparse)[d];
      DataType feature_dtype =
          is_ragged ? config.ragged[d].dtype : config.sparse[d].dtype;
      if (example_dtype != DT_INVALID && example_dtype != feature_dtype) {
        return example_error(
            strings::StrCat("Data types don't match. ",
                            "Expected type: ", DataTypeString(feature_dtype),
                            ", Actual type: ", DataTypeString(example_dtype)));
      }

      switch (feature_dtype) {
        case DT_INT64: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseInt64List(&out.int64_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.int64_list.size());
          break;
        }
        case DT_FLOAT: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseFloatList(&out.float_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.float_list.size());
          break;
        }
        case DT_STRING: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseBytesList(&out.bytes_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.bytes_list.size());
          break;
        }
        default:
          LOG(FATAL) << "Should not happen.";
      }

      if (output_stats) {
        // Use `out.example_end_indices` to determine the feature-value count
        // for this feature, because the preceding switch statement pushes
        // the length of the appropriate feature list to that vector.
        // TODO(b/111553342): If desirable, we could add support for counting
        // elements in the features that aren't parsed, but this could add
        // considerable runtime cost.
        const size_t out_examples_count = out.example_end_indices.size();
        if (out_examples_count == 1) {
          output_stats->feature_values_count += out.example_end_indices[0];
        } else {
          output_stats->feature_values_count +=
              out.example_end_indices[out_examples_count - 1] -
              out.example_end_indices[out_examples_count - 2];
        }
      }
    }
  }

  // Handle missing dense features for fixed strides.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (config.dense[d].variable_length) continue;
    if (dense_feature_last_example[d] == example_index) continue;
    if (config.dense[d].default_value.NumElements() == 0) {
      return errors::InvalidArgument(
          "Name: ", example_name, ", Feature: ", config.dense[d].feature_name,
          " (data type: ", DataTypeString(config.dense[d].dtype), ")",
          " is required but could not be found.");
    }
    const Tensor& in = config.dense[d].default_value;
    Tensor& out = (*output_dense)[d];
    const std::size_t num_elements = in.shape().num_elements();
    const std::size_t offset = example_index * num_elements;

    switch (config.dense[d].dtype) {
      case DT_INT64: {
        std::copy_n(in.flat<int64>().data(), num_elements,
                    out.flat<int64>().data() + offset);
        break;
      }
      case DT_FLOAT: {
        std::copy_n(in.flat<float>().data(), num_elements,
                    out.flat<float>().data() + offset);
        break;
      }
      case DT_STRING: {
        std::copy_n(in.flat<tstring>().data(), num_elements,
                    out.flat<tstring>().data() + offset);
        break;
      }
      default:
        LOG(FATAL) << "Should not happen.";
    }
  }

  // Handle missing varlen dense features.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (!config.dense[d].variable_length) continue;
    if (dense_feature_last_example[d] == example_index) continue;
    SparseBuffer& out = (*output_varlen_dense)[d];
    size_t prev_example_end_index =
        out.example_end_indices.empty() ? 0 : out.example_end_indices.back();
    out.example_end_indices.push_back(prev_example_end_index);
  }

  // Handle missing sparse features.
  for (size_t d = 0; d < config.sparse.size(); ++d) {
    if (sparse_feature_last_example[d] == example_index) continue;
    SparseBuffer& out = (*output_sparse)[d];
    size_t prev_example_end_index =
        out.example_end_indices.empty() ? 0 : out.example_end_indices.back();
    out.example_end_indices.push_back(prev_example_end_index);
