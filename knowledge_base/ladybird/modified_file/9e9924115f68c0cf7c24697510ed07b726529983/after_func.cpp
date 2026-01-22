    }
    return page.release_nonnull();
}

ErrorOr<NonnullRefPtr<PhysicalPage>> MemoryManager::allocate_physical_page(ShouldZeroFill should_zero_fill, bool* did_purge)
{
    SpinlockLocker lock(s_mm_lock);
    auto page = find_free_physical_page(false);
    bool purged_pages = false;

    if (!page) {
        // We didn't have a single free physical page. Let's try to free something up!
        // First, we look for a purgeable VMObject in the volatile state.
        for_each_vmobject([&](auto& vmobject) {
            if (!vmobject.is_anonymous())
                return IterationDecision::Continue;
            auto& anonymous_vmobject = static_cast<AnonymousVMObject&>(vmobject);
            if (!anonymous_vmobject.is_purgeable() || !anonymous_vmobject.is_volatile())
                return IterationDecision::Continue;
            if (auto purged_page_count = anonymous_vmobject.purge()) {
                dbgln("MM: Purge saved the day! Purged {} pages from AnonymousVMObject", purged_page_count);
                page = find_free_physical_page(false);
                purged_pages = true;
                VERIFY(page);
                return IterationDecision::Break;
            }
            return IterationDecision::Continue;
        });
        // Second, we look for a file-backed VMObject with clean pages.
        for_each_vmobject([&](auto& vmobject) {
            if (!vmobject.is_inode())
                return IterationDecision::Continue;
            auto& inode_vmobject = static_cast<InodeVMObject&>(vmobject);
            // FIXME: It seems excessive to release *all* clean pages from the inode when we only need one.
            if (auto released_page_count = inode_vmobject.release_all_clean_pages()) {
                dbgln("MM: Clean inode release saved the day! Released {} pages from InodeVMObject", released_page_count);
                page = find_free_physical_page(false);
                VERIFY(page);
                return IterationDecision::Break;
            }
            return IterationDecision::Continue;
        });
        if (!page) {
            dmesgln("MM: no physical pages available");
            return ENOMEM;
        }
    }

    if (should_zero_fill == ShouldZeroFill::Yes) {
        auto* ptr = quickmap_page(*page);
        memset(ptr, 0, PAGE_SIZE);
        unquickmap_page();
    }

