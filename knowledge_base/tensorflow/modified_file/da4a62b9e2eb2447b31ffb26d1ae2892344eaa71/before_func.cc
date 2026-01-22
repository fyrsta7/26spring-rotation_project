//     c. For all t: sum_{i in L[t]} s[i]^T * m[i] <= M
//       Make sure e is one-hot:
//     d. For all (i, j) in E, e[i, j] in {0, 1} ^ dim(e[i, j])
//     e. For all (i, j) in E, e[i, j]^T * 1 == 1
//       Make sure s[i] and s[j] align with e[i, j]:
//     f. For all (i, j) in E, 0 <= p < dim(s[i]),
//        sum_{0 <= q < dim(s[j])} e[i, j](p * dim(s[j]) + q) <= s[i](p)
//     g. For all (i, j) in E, 0 <= q < dim(s[j]),
//        sum_{0 <= p < dim(s[i])} e[i, j](p * dim(s[j]) + q) <= s[j](q)
//     h. For all (i, j) in A and all (p, q),
//        s[i][p] + s[j][q] <= 1 if v[p, q] == 1.0
// Serialize parameters of the ILP problem as numpy arrays and call the python
// solver.
ORToolsSolverResult CallORToolsSolver(
    int64_t N, int64_t M, const std::vector<int>& s_len,
    const std::vector<int>& s_follow, const std::vector<std::pair<int, int>>& E,
    const std::vector<std::vector<int>>& L,
    const std::vector<std::vector<double>>& c,
    const std::vector<std::vector<double>>& d,
    const std::vector<std::vector<double>>& m,
    const std::vector<std::vector<double>>& r,
    const std::vector<std::pair<int, int>>& A,
    const std::vector<std::vector<double>>& v,
    const std::vector<std::string>& instruction_names,
    int64_t solver_timeout_in_seconds, bool crash_at_infinity_costs_check) {
  size_t num_edges = E.size();

  int32_t num_workers = 32;
  // SAT or SCIP
  std::unique_ptr<MPSolver> solver(std::make_unique<MPSolver>("", MPSolver::SAT_INTEGER_PROGRAMMING));
  CHECK(solver);
  solver->MutableObjective()->SetMinimization();
  std::string solver_parameter_str;
#ifdef PLATFORM_GOOGLE
  if (solver->ProblemType() ==
      operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
    // Set random_seed, interleave_search and share_binary_clauses for
    // determinism, and num_workers for parallelism.
    solver_parameter_str = absl::StrCat(
        "share_binary_clauses:false,random_seed:1,interleave_"
        "search:true,num_workers:",
        num_workers);
    solver->SetSolverSpecificParametersAsString(solver_parameter_str);
  }
#endif
  // Create variables
  std::vector<std::vector<MPVariable*>> s(N);
  std::vector<std::vector<MPVariable*>> e(num_edges);

  size_t var_vector_cnt = 0;
  for (size_t i = 0; i < N; ++i) {
    if (s_follow[i] < 0) {
      var_vector_cnt += 1;
      // Creates variables for instructions that do not follow others.
      solver->MakeBoolVarArray(s_len[i], absl::StrCat("s[", i, "]"), &s[i]);
    }
  }

  for (size_t i = 0; i < N; ++i) {
    if (s_follow[i] >= 0) {
      // Copies the variable of followed instruction to the following
      // instruction.
      s[i] = s[s_follow[i]];
    }
  }

  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = E[i];
    solver->MakeBoolVarArray(
        s_len[edge.first] * s_len[edge.second],
        absl::StrCat("e[", edge.first, ",", edge.second, "]"), &e[i]);
  }

  // Objective
  // Node costs
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < s[i].size(); ++j) {
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(s[i][j]);
      solver->MutableObjective()->SetCoefficient(
          s[i][j], accumulated_coefficient + c[i][j] + d[i][j]);
    }
  }
  // Edge costs
  for (size_t i = 0; i < num_edges; ++i) {
    for (size_t j = 0; j < e[i].size(); ++j) {
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(e[i][j]);
      solver->MutableObjective()->SetCoefficient(
          e[i][j], accumulated_coefficient + r[i][j]);
    }
  }

  // Constraints
  // 0. Do not choose solutions with infinity costs, as it will make the
  // objective value so large that other solution choices do not matter anymore.
  // Remove these constraints once b/238210866 is done.
  for (size_t i = 0; i < N; ++i) {
    if (s[i].empty()) {
      continue;
    }
    bool all_infinity = true;
    for (size_t j = 0; j < s[i].size(); ++j) {
      if (solver->MutableObjective()->GetCoefficient(s[i][j]) >=
          kInfinityCost) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0, absl::StrCat("infinitycost: s[", i, "][", j, "] = 0"));
        constraint->SetCoefficient(s[i][j], 1.0);
      } else {
        all_infinity = false;
      }
    }
    if (all_infinity) {
      LOG(FATAL) << "All of s[" << i << "][*] have infinity costs";
    }
  }

  for (size_t i = 0; i < num_edges; ++i) {
    if (e[i].empty()) {
      continue;
    }
    bool all_infinity = true;
    for (size_t j = 0; j < e[i].size(); ++j) {
      std::pair<int, int> edge = E[i];
      solver->MutableObjective()->SetCoefficient(e[i][j], r[i][j]);
      if (r[i][j] >= kInfinityCost) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0,
            absl::StrCat("infinitycost: e[", edge.first, "][", edge.second,
                         "][", j, "] = 0"));
        constraint->SetCoefficient(e[i][j], 1.0);
      } else {
        all_infinity = false;
      }
    }
    if (all_infinity) {
      auto err_msg = absl::StrCat("All of e[", E[i].first, "][", E[i].second,
                                  "][*] have infinity costs");
      if (crash_at_infinity_costs_check) {
        LOG(FATAL) << err_msg;
      } else {
        LOG(WARNING) << err_msg;
        return ORToolsSolverResult(absl::InternalError(err_msg), false);
      }
    }
  }

  // a. specified via "BoolVarArray"
  // b.
  for (size_t i = 0; i < N; ++i) {
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(s[", i, "][j] for j = [0 .. ", s[i].size(),
                     ")) = 1"));
    for (size_t j = 0; j < s[i].size(); ++j) {
      constraint->SetCoefficient(s[i][j], 1.0);
    }
  }
  // c.
  if (M > 0) {
    for (size_t t = 0; t < L.size(); ++t) {
      std::string str = "[";
      for (auto i : L[t]) {
        absl::StrAppend(&str, i, ", ");
      }
      str += "]";
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), M, absl::StrCat("mem[", t, "] = ", str));
      for (auto i : L[t]) {
        for (size_t j = 0; j < s[i].size(); ++j) {
          double accumulated_coefficient = constraint->GetCoefficient(s[i][j]);
          constraint->SetCoefficient(s[i][j],
                                     accumulated_coefficient + m[i][j]);
        }
      }
    }
  }

  // d. specified via "BoolVarArray"
  // e.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = E[i];
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(e[", edge.first, "][", edge.second, "][*]) = 1"));
    for (size_t j = 0; j < e[i].size(); ++j) {
      constraint->SetCoefficient(e[i][j], 1.0);
    }
  }
  // f.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = E[i];
    for (size_t p = 0; p < s[edge.first].size(); ++p) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0, absl::StrCat("f for i = ", i, ", p = ", p));
      constraint->SetCoefficient(s[edge.first][p], -1.0);
      for (size_t q = 0; q < s[edge.second].size(); ++q) {
        constraint->SetCoefficient(e[i][p * s[edge.second].size() + q], 1.0);
      }
    }
  }
  // g.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = E[i];
    for (size_t q = 0; q < s[edge.second].size(); ++q) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0, absl::StrCat("g for i = ", i, ", q = ", q));
      constraint->SetCoefficient(s[edge.second][q], -1.0);
      for (size_t p = 0; p < s[edge.first].size(); ++p) {
        constraint->SetCoefficient(e[i][p * s[edge.second].size() + q], 1.0);
      }
    }
  }
  // h.
  for (size_t i = 0; i < A.size(); ++i) {
    std::pair<int, int> alias = A[i];
    for (size_t p = 0; p < s[alias.first].size(); ++p) {
      for (size_t q = 0; q < s[alias.second].size(); ++q) {
        // if lhs == 1
        if (v[i][p * s[alias.second].size() + q] > 0.5) {
          MPConstraint* constraint = solver->MakeRowConstraint(
              -MPSolver::infinity(), 1,
              absl::StrCat("s[", alias.first, "][", p, "] + s[", alias.second,
                           "][", q, "] <= 1"));
          constraint->SetCoefficient(s[alias.first][p], 1.0);
          constraint->SetCoefficient(s[alias.second][q], 1.0);
        }
      }
    }
  }

#ifdef PLATFORM_GOOGLE
  // Exports the model for debugging.
  bool dump_model = false;
  if (dump_model) {
    operations_research::MPModelProto model_proto;
    solver->ExportModelToProto(&model_proto);
    auto write_status = file::SetTextProto(
        // Modify this file path if needed.
        absl::StrCat("/tmp/model_", solver->NumVariables(), ".proto"),
        model_proto, file::Defaults());
    if (!write_status.ok()) {
      LOG(ERROR) << write_status.message();
    }
  }
#endif
  solver->SetTimeLimit(absl::Seconds(solver_timeout_in_seconds));
  VLOG(0) << "Starting solver " << solver->ProblemType() << "\n"
          << "Solver parameter string: " << solver_parameter_str << "\n"
          << "Number of workers: " << num_workers << "\n"
          << "Number of threads: " << solver->GetNumThreads() << "\n"
          << "Time limit: " << solver->time_limit() << "\n"
          << "Number variables for ILP: " << solver->NumVariables() << "\n"
          << "Total vector of variables: " << var_vector_cnt << "\n"
          << "Total instructions: " << N << "\n"
          << "Memory budget: " << M / (1024 * 1024 * 1024) << "GB\n"
          << "Number of ILP constraints: " << solver->NumConstraints();
  auto status = solver->Solve();

  if (status == operations_research::MPSolver::INFEASIBLE) {
    LOG(ERROR) << "MPSolver could not find any feasible solution.";
#ifdef PLATFORM_GOOGLE
    operations_research::MPModelRequest model_request;
    solver->ExportModelToProto(model_request.mutable_model());
    if (solver->ProblemType() ==
        operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
      model_request.set_solver_type(
          operations_research::MPModelRequest::SAT_INTEGER_PROGRAMMING);
    } else if (solver->ProblemType() ==
               operations_research::MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING) {
      model_request.set_solver_type(
          operations_research::MPModelRequest::SCIP_MIXED_INTEGER_PROGRAMMING);
    }
    model_request.set_solver_time_limit_seconds(100);
    auto iis = MPSolver::ComputeIrreducibleInfeasibleSubset(model_request);
    LOG(INFO) << iis.status().DebugString();
    LOG(INFO) << "Infeasible constraints: ";
    for (int index : iis.constraint_index()) {
      LOG(INFO) << " - " << model_request.model().constraint(index).name();
    }
    for (int index : iis.general_constraint_index()) {
      LOG(INFO)
          << " - "
          << model_request.model().general_constraint(index).DebugString();
    }
#endif

    return ORToolsSolverResult(
        absl::InternalError("MPSolver could not find any feasible solution."),
        false);
  } else if (status != operations_research::MPSolver::OPTIMAL) {
    auto err_msg = "Solver timed out. Will proceed without auto sharding.";
    LOG(WARNING) << err_msg;

    // The solver timed out. We now rely on heuristic-based sharding propagation
    // to degrade gracefully.
    return ORToolsSolverResult(absl::InternalError(err_msg), true);
  }

  LOG(INFO) << "Solver Status: " << status
            << " Objective value: " << solver->Objective().Value();
  if (solver->Objective().Value() >= kInfinityCost) {
    LOG(WARNING) << "Objective (" << solver->Objective().Value()
                 << ") is larger than kInfinityCost. It means the solver "
                    "chooses a solution with kInfinityCost and there may be "
                    "numerical issues when the solver considering other costs.";
  }
  if (VLOG_IS_ON(10)) {
    // Print solver information for debugging. This hasn't been useful so far,
    // so leave it at VLOG level 10.
    operations_research::MPModelProto model_proto;
    solver->ExportModelToProto(&model_proto);
    VLOG(10) << "MODEL:";
    XLA_VLOG_LINES(10, model_proto.DebugString());
    VLOG(10) << "RESPONSE:";
    operations_research::MPSolutionResponse response;
    solver->FillSolutionResponseProto(&response);
    XLA_VLOG_LINES(10, response.DebugString());
  }

  // Return value
  std::vector<int64_t> chosen_strategy(N, -1), e_val(num_edges, -1);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < s[i].size(); ++j) {
      // if lhs == 1
      if (s[i][j]->solution_value() > 0.5) {
        chosen_strategy[i] = j;
        break;
      }
    }
  }
  for (int i = 0; i < num_edges; ++i) {
    for (int j = 0; j < e[i].size(); ++j) {
      // if lhs == 1
      if (e[i][j]->solution_value() > 0.5) {
        e_val[i] = j;
        break;
      }
    }
  }
