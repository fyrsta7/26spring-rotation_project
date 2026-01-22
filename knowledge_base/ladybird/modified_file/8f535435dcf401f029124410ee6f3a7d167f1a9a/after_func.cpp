{
    Accessor* accessor { nullptr };
    auto property_metadata = shape().lookup(property_name.to_string_or_symbol());
    if (property_metadata.has_value()) {
        auto existing_property = get_direct(property_metadata.value().offset);
        if (existing_property.is_accessor())
            accessor = &existing_property.as_accessor();
    }
    if (!accessor) {
        accessor = Accessor::create(vm(), nullptr, nullptr);
        bool definition_success = define_property(property_name, accessor, attributes, throw_exceptions);
        if (vm().exception())
            return {};
        if (!definition_success)
            return false;
    }
    if (is_getter)
        accessor->set_getter(&getter_or_setter);
    else
        accessor->set_setter(&getter_or_setter);

    return true;
}

bool Object::put_own_property(Object& this_object, const StringOrSymbol& property_name, Value value, PropertyAttributes attributes, PutOwnPropertyMode mode, bool throw_exceptions)
{
    ASSERT(!(mode == PutOwnPropertyMode::Put && value.is_accessor()));

    if (value.is_accessor()) {
        auto& accessor = value.as_accessor();
        if (accessor.getter())
            attributes.set_has_getter();
        if (accessor.setter())
            attributes.set_has_setter();
    }

    // NOTE: We disable transitions during initialize(), this makes building common runtime objects significantly faster.
    //       Transitions are primarily interesting when scripts add properties to objects.
    if (!m_transitions_enabled && !m_shape->is_unique()) {
        m_shape->add_property_without_transition(property_name, attributes);
        m_storage.resize(m_shape->property_count());
        m_storage[m_shape->property_count() - 1] = value;
        return true;
    }

    auto metadata = shape().lookup(property_name);
    bool new_property = !metadata.has_value();

    if (!is_extensible() && new_property) {
#ifdef OBJECT_DEBUG
        dbg() << "Disallow define_property of non-extensible object";
#endif
        if (throw_exceptions && vm().in_strict_mode())
            vm().throw_exception<TypeError>(global_object(), ErrorType::NonExtensibleDefine, property_name.to_display_string());
        return false;
    }

    if (new_property) {
        if (!m_shape->is_unique() && shape().property_count() > 100) {
            // If you add more than 100 properties to an object, let's stop doing
            // transitions to avoid filling up the heap with shapes.
            ensure_shape_is_unique();
        }

        if (m_shape->is_unique()) {
            m_shape->add_property_to_unique_shape(property_name, attributes);
            m_storage.resize(m_shape->property_count());
        } else if (m_transitions_enabled) {
            set_shape(*m_shape->create_put_transition(property_name, attributes));
        } else {
            m_shape->add_property_without_transition(property_name, attributes);
            m_storage.resize(m_shape->property_count());
        }
        metadata = shape().lookup(property_name);
        ASSERT(metadata.has_value());
    }

    if (!new_property && mode == PutOwnPropertyMode::DefineProperty && !metadata.value().attributes.is_configurable() && attributes != metadata.value().attributes) {
#ifdef OBJECT_DEBUG
        dbg() << "Disallow reconfig of non-configurable property";
#endif
        if (throw_exceptions)
            vm().throw_exception<TypeError>(global_object(), ErrorType::DescChangeNonConfigurable, property_name.to_display_string());
        return false;
    }

    if (mode == PutOwnPropertyMode::DefineProperty && attributes != metadata.value().attributes) {
        if (m_shape->is_unique()) {
            m_shape->reconfigure_property_in_unique_shape(property_name, attributes);
        } else {
            set_shape(*m_shape->create_configure_transition(property_name, attributes));
        }
        metadata = shape().lookup(property_name);
