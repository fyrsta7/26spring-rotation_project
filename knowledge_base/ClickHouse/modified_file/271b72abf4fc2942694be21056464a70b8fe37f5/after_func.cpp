    bool tryRemoveSorting(QueryPlan::Node * sorting_node, QueryPlan::Node * parent_node)
    {
        if (!canRemoveCurrentSorting())
            return false;

        /// remove sorting
        for (auto & child : parent_node->children)
        {
            if (child == sorting_node)
            {
                child = sorting_node->children.front();
                break;
            }
        }

        /// sorting removed, so need to update sorting traits for upstream steps
        const DataStream * input_stream = &parent_node->children.front()->step->getOutputStream();
        chassert(parent_node == (stack.rbegin() + 1)->node); /// skip element on top of stack since it's sorting which was just removed
        for (StackWithParent::const_reverse_iterator it = stack.rbegin() + 1; it != stack.rend(); ++it)
        {
            const QueryPlan::Node * node = it->node;
            /// skip removed sorting steps
            auto * step = node->step.get();
            if (typeid_cast<const SortingStep *>(step) && node != nodes_affect_order.back())
                continue;

            logStep("update sorting traits", node);

            auto * trans = dynamic_cast<ITransformingStep *>(step);
            if (!trans)
            {
                logStep("stop update sorting traits: node is not transforming step", node);
                break;
            }

            trans->updateInputStream(*input_stream);
            input_stream = &trans->getOutputStream();

            /// update sorting properties though stack until reach node which affects order (inclusive)
            if (node == nodes_affect_order.back())
            {
                logStep("stop update sorting traits: reached node which affect order", node);
                break;
            }
        }

        return true;
    }
