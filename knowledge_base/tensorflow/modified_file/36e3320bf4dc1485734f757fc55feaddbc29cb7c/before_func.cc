      for (int j = 0; j < number_attr; j++) {
        (*fdef->mutable_ret())[signature.output_arg(next_sig_output++).name()] =
            absl::StrCat(node_name, ":", output_arg.name(), ":", j);
      }
    } else {
      (*fdef->mutable_ret())[signature.output_arg(next_sig_output++).name()] =
          absl::StrCat(node_name, ":", output_arg.name(), ":0");
