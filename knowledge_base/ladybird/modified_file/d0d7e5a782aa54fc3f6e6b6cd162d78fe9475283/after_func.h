
template<typename NodeType>
class NonElementParentNode {
public:
    JS::GCPtr<Element const> get_element_by_id(FlyString const& id) const
    {
        JS::GCPtr<Element const> found_element;
        static_cast<NodeType const*>(this)->template for_each_in_inclusive_subtree_of_type<Element>([&](auto& element) {
            if (element.id() == id) {
                found_element = &element;
                return IterationDecision::Break;
            }
