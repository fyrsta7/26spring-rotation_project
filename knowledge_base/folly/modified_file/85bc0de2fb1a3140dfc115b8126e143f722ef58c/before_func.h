        }
      }
      tracking->lifetimes.erase(this);
    }

    void track(CounterAndCache& state) {
      auto& global = Global::instance();
      state.cache = &state.counter;
      auto const tracking = global.tracking.wlock();
      auto const inserted = tracking->lifetimes[this].insert(&state.counter);
      tracking->locals[&state.counter] += inserted.second;
