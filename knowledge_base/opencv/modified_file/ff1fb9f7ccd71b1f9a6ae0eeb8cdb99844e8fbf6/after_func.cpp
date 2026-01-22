            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance)
            {
                matches_info.matches.push_back(m0);
                matches.insert(make_pair(m0.queryIdx, m0.trainIdx));
            }
        }

        // Find 2->1 matches
        pair_matches.clear();
        matcher.knnMatch(descriptors2_, descriptors1_, train_idx_, distance_, all_dist_, 2);
        matcher.knnMatchDownload(train_idx_, distance_, pair_matches);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
