void ConnectionManager::addOrUpdateGroupMember(absl::string_view group,
                                               absl::string_view client_id) {
  ENVOY_LOG(trace, "#addOrUpdateGroupMember. Group: {}, client ID: {}", group, client_id);
  auto search = group_members_.find(group);
  if (search == group_members_.end()) {
    std::vector<ConsumerGroupMember> members;
    members.emplace_back(ConsumerGroupMember(client_id, *this));
    group_members_.emplace(std::string(group.data(), group.size()), members);
  } else {
    std::vector<ConsumerGroupMember>& members = search->second;
    for (auto it = members.begin(); it != members.end();) {
      if (it->clientId() == client_id) {
        it->refresh();
        ++it;
      } else if (it->expired()) {
        it = members.erase(it);
      } else {
        ++it;
      }
    }
    if (members.empty()) {
      group_members_.erase(search);
    }
  }
}
