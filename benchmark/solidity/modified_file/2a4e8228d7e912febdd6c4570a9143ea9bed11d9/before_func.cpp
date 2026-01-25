BOOST_AUTO_TEST_CASE(trie_tests)
{
	string testPath = test::getTestPath();


	testPath += "/TrieTests";

	cnote << "Testing Trie...";
	js::mValue v;
	string s = asString(contents(testPath + "/trietest.json"));
	BOOST_REQUIRE_MESSAGE(s.length() > 0, "Contents of 'trietest.json' is empty. Have you cloned the 'tests' repo branch develop?");
	js::read_string(s, v);
	for (auto& i: v.get_obj())
	{
		cnote << i.first;
		js::mObject& o = i.second.get_obj();
		vector<pair<string, string>> ss;
		for (auto i: o["in"].get_array())
		{
			vector<string> values;
			for (auto s: i.get_array())
				values.push_back(s.get_str());

			assert(values.size() == 2);
			ss.push_back(make_pair(values[0], values[1]));
			if (!ss.back().first.find("0x"))
				ss.back().first = asString(fromHex(ss.back().first.substr(2)));
			if (!ss.back().second.find("0x"))
				ss.back().second = asString(fromHex(ss.back().second.substr(2)));
		}
		for (unsigned j = 0; j < min(1000u, dev::test::fac((unsigned)ss.size())); ++j)
		{
			MemoryDB m;
			GenericTrieDB<MemoryDB> t(&m);
			t.init();
			BOOST_REQUIRE(t.check(true));
			for (auto const& k: ss)
			{
				t.insert(k.first, k.second);
				BOOST_REQUIRE(t.check(true));
			}
			BOOST_REQUIRE(!o["root"].is_null());
			BOOST_CHECK_EQUAL(o["root"].get_str(), "0x" + toHex(t.root().asArray()));
			if (o["root"].get_str() != "0x" + toHex(t.root().asArray()))
				break;
		}
	}
}