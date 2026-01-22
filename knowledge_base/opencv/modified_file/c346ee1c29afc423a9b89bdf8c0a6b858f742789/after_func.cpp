
                    const int k = neighbor_idx & 3;
                    CV_DbgAssert(q);
                    if (!q->neighbors[k])
                    {
                        if (normL2Sqr<float>(closest_corner.pt - q->corners[k]->pt) < min_dist)
                            break;
                    }
                }
                if (neighbor_idx_idx < neighbors_count)
                    continue;

                closest_corner.pt = (pt + closest_corner.pt) * 0.5f;

                // We've found one more corner - remember it
                cur_quad.count++;
                cur_quad.neighbors[i] = closest_quad;
                cur_quad.corners[i] = &closest_corner;

                closest_quad->count++;
                closest_quad->neighbors[closest_corner_idx] = &cur_quad;
            }
        }
    }
}


// returns corners in clockwise order
// corners don't necessarily start at same position on quad (e.g.,
//   top left corner)
void ChessBoardDetector::generateQuads(const cv::Mat& image_, int flags, int dilations)
{
    binarized_image = image_;  // save for debug purposes

    int quad_count = 0;

    all_quads.deallocate();
    all_corners.deallocate();

    // empiric bound for minimal allowed area for squares
    const int min_area = 25; //cvRound( image->cols * image->rows * .03 * 0.01 * 0.92 );

    bool filterQuads = (flags & CALIB_CB_FILTER_QUADS) != 0;

    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    cv::findContours(image_, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        CV_LOG_DEBUG(NULL, "calib3d(chessboard): cv::findContours() returns no contours");
        return;
    }

    std::vector<int> contour_child_counter(contours.size(), 0);
    int boardIdx = -1;

    std::vector<QuadCountour> contour_quads;

    for (int idx = (int)(contours.size() - 1); idx >= 0; --idx)
    {
        int parentIdx = hierarchy[idx][3];
        if (hierarchy[idx][2] != -1 || parentIdx == -1)  // holes only (no child contours and with parent)
            continue;
        const std::vector<Point>& contour = contours[idx];

        Rect contour_rect = boundingRect(contour);
        if (contour_rect.area() < min_area)
            continue;

        std::vector<Point> approx_contour = contour;

        const int min_approx_level = 1, max_approx_level = MAX_CONTOUR_APPROX;
        for (int approx_level = min_approx_level; approx_contour.size() > 4 && approx_level <= max_approx_level; approx_level++ )
        {
            approxPolyDP(approx_contour, approx_contour, (float)approx_level, true);
        }

        // reject non-quadrangles
        if (approx_contour.size() != 4)
            continue;
        if (!cv::isContourConvex(approx_contour))
            continue;

        cv::Point pt[4];
        for (int i = 0; i < 4; ++i)
            pt[i] = approx_contour[i];
        CV_LOG_VERBOSE(NULL, 9, "... contours(" << contour_quads.size() << " added):" << pt[0] << " " << pt[1] << " " << pt[2] << " " << pt[3]);

        if (filterQuads)
        {
            double p = cv::arcLength(approx_contour, true);
            double area = cv::contourArea(approx_contour, false);

            double d1 = sqrt(normL2Sqr<double>(pt[0] - pt[2]));
            double d2 = sqrt(normL2Sqr<double>(pt[1] - pt[3]));

            // philipg.  Only accept those quadrangles which are more square
            // than rectangular and which are big enough
            double d3 = sqrt(normL2Sqr<double>(pt[0] - pt[1]));
            double d4 = sqrt(normL2Sqr<double>(pt[1] - pt[2]));
            if (!(d3*4 > d4 && d4*4 > d3 && d3*d4 < area*1.5 && area > min_area &&
                d1 >= 0.15 * p && d2 >= 0.15 * p))
                continue;
        }

        contour_child_counter[parentIdx]++;
        if (boardIdx != parentIdx && (boardIdx < 0 || contour_child_counter[boardIdx] < contour_child_counter[parentIdx]))
            boardIdx = parentIdx;

        contour_quads.emplace_back(pt, parentIdx);
    }

    size_t total = contour_quads.size();
    size_t max_quad_buf_size = std::max((size_t)2, total * 3);
    all_quads.allocate(max_quad_buf_size);
    all_corners.allocate(max_quad_buf_size * 4);

    // Create array of quads structures
    for (size_t idx = 0; idx < total; ++idx)
    {
        QuadCountour& qc = contour_quads[idx];
        if (filterQuads && qc.parent_contour != boardIdx)
            continue;

