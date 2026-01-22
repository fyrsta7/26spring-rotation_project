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

