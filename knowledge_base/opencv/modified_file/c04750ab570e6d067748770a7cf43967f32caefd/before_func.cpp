    Mat postProcess(const std::vector<Mat>& output_blobs)
    {
        Mat faces;
        for (size_t i = 0; i < strides.size(); ++i) {
            int cols = int(padW / strides[i]);
            int rows = int(padH / strides[i]);

            // Extract from output_blobs
            Mat cls = output_blobs[i];
            Mat obj = output_blobs[i + strides.size() * 1];
            Mat bbox = output_blobs[i + strides.size() * 2];
            Mat kps = output_blobs[i + strides.size() * 3];

            // Decode from predictions
            float* cls_v = (float*)(cls.data);
            float* obj_v = (float*)(obj.data);
            float* bbox_v = (float*)(bbox.data);
            float* kps_v = (float*)(kps.data);

            // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
            // 'tl': top left point of the bounding box
            // 're': right eye, 'le': left eye
            // 'nt':  nose tip
            // 'rcm': right corner of mouth, 'lcm': left corner of mouth
            Mat face(1, 15, CV_32FC1);

            for(int r = 0; r < rows; ++r) {
                for(int c = 0; c < cols; ++c) {
                    size_t idx = r * cols + c;

                    // Get score
                    float cls_score = cls_v[idx];
                    float obj_score = obj_v[idx];

                    // Clamp
                    cls_score = MIN(cls_score, 1.f);
                    cls_score = MAX(cls_score, 0.f);
                    obj_score = MIN(obj_score, 1.f);
                    obj_score = MAX(obj_score, 0.f);
                    float score = std::sqrt(cls_score * obj_score);
                    face.at<float>(0, 14) = score;

                    // Get bounding box
                    float cx = ((c + bbox_v[idx * 4 + 0]) * strides[i]);
                    float cy = ((r + bbox_v[idx * 4 + 1]) * strides[i]);
                    float w = exp(bbox_v[idx * 4 + 2]) * strides[i];
                    float h = exp(bbox_v[idx * 4 + 3]) * strides[i];

                    float x1 = cx - w / 2.f;
                    float y1 = cy - h / 2.f;

                    face.at<float>(0, 0) = x1;
                    face.at<float>(0, 1) = y1;
                    face.at<float>(0, 2) = w;
                    face.at<float>(0, 3) = h;

                    // Get landmarks
                    for(int n = 0; n < 5; ++n) {
                        face.at<float>(0, 4 + 2 * n) = (kps_v[idx * 10 + 2 * n] + c) * strides[i];
                        face.at<float>(0, 4 + 2 * n + 1) = (kps_v[idx * 10 + 2 * n + 1]+ r) * strides[i];
                    }
                    faces.push_back(face);
                }
            }
        }

        if (faces.rows > 1)
        {
            // Retrieve boxes and scores
            std::vector<Rect2i> faceBoxes;
            std::vector<float> faceScores;
            for (int rIdx = 0; rIdx < faces.rows; rIdx++)
            {
                faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
                                           int(faces.at<float>(rIdx, 1)),
                                           int(faces.at<float>(rIdx, 2)),
                                           int(faces.at<float>(rIdx, 3))));
                faceScores.push_back(faces.at<float>(rIdx, 14));
            }

            std::vector<int> keepIdx;
            dnn::NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

            // Get NMS results
            Mat nms_faces;
            for (int idx: keepIdx)
            {
                nms_faces.push_back(faces.row(idx));
            }
            return nms_faces;
        }
        else
        {
            return faces;
        }
    }
