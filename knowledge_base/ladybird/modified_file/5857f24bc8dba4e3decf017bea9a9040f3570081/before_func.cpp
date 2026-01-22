    }

    // FIXME: This should also be a function of the animation-timing-function, if not during the delay.
    return output_progress;
}

void StyleComputer::ensure_animation_timer() const
{
    constexpr static auto timer_delay_ms = 1000 / 60;
    if (!m_animation_driver_timer) {
        m_animation_driver_timer = Platform::Timer::create_repeating(timer_delay_ms, [this] {
            // If we run out of animations, stop the timer - it'll turn back on the next time we have an active animation.
            if (m_active_animations.is_empty()) {
                m_animation_driver_timer->stop();
                return;
            }

            HashTable<AnimationKey> animations_to_remove;
            HashTable<DOM::Element*> owning_elements_to_invalidate;

            for (auto& it : m_active_animations) {
                if (!it.value->owning_element) {
                    // The element disappeared since we last ran, just discard the animation.
                    animations_to_remove.set(it.key);
                    continue;
                }

                auto transition = it.value->step(CSS::Time { timer_delay_ms, CSS::Time::Type::Ms });
                owning_elements_to_invalidate.set(it.value->owning_element);

                switch (transition) {
                case AnimationStepTransition::NoTransition:
                    break;
                case AnimationStepTransition::IdleOrBeforeToActive:
                    // FIXME: Dispatch `animationstart`.
                    break;
                case AnimationStepTransition::IdleOrBeforeToAfter:
                    // FIXME: Dispatch `animationstart` then `animationend`.
                    m_finished_animations.set(it.key, move(it.value->active_state_if_fill_forward));
                    break;
                case AnimationStepTransition::ActiveToBefore:
                    // FIXME: Dispatch `animationend`.
                    m_finished_animations.set(it.key, move(it.value->active_state_if_fill_forward));
                    break;
                case AnimationStepTransition::ActiveToActiveChangingTheIteration:
                    // FIXME: Dispatch `animationiteration`.
                    break;
                case AnimationStepTransition::ActiveToAfter:
                    // FIXME: Dispatch `animationend`.
                    m_finished_animations.set(it.key, move(it.value->active_state_if_fill_forward));
                    break;
                case AnimationStepTransition::AfterToActive:
                    // FIXME: Dispatch `animationstart`.
                    break;
                case AnimationStepTransition::AfterToBefore:
                    // FIXME: Dispatch `animationstart` then `animationend`.
                    m_finished_animations.set(it.key, move(it.value->active_state_if_fill_forward));
                    break;
                case AnimationStepTransition::Cancelled:
                    // FIXME: Dispatch `animationcancel`.
                    m_finished_animations.set(it.key, nullptr);
                    break;
                }
                if (it.value->is_done())
                    animations_to_remove.set(it.key);
            }

            for (auto key : animations_to_remove)
                m_active_animations.remove(key);

            for (auto* element : owning_elements_to_invalidate)
