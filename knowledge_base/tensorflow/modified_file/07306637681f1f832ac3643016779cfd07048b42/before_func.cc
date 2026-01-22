              decoded_boxes)[selected_index];
      // detection_classes
      GetTensorData<float>(detection_classes)[box_offset] = class_indices[col];
      // detection_scores
      GetTensorData<float>(detection_scores)[box_offset] =
          box_scores[class_indices[col]];
    }
    output_box_index++;
  }
  GetTensorData<float>(num_detections)[0] = output_box_index;
  return kTfLiteOk;
}

