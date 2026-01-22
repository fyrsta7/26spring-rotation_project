
#include <tools/yulPhaser/Random.h>

#include <boost/test/unit_test.hpp>

#include <cassert>

using namespace std;

namespace solidity::phaser::test
{

BOOST_AUTO_TEST_SUITE(Phaser)
BOOST_AUTO_TEST_SUITE(RandomTest)

BOOST_AUTO_TEST_CASE(uniformRandomInt_returns_different_values_when_called_multiple_times)
{
	constexpr uint32_t numSamples = 1000;
	constexpr uint32_t numOutcomes = 100;

	vector<uint32_t> samples1;
	vector<uint32_t> samples2;
	for (uint32_t i = 0; i < numSamples; ++i)
	{
		samples1.push_back(uniformRandomInt(0, numOutcomes - 1));
		samples2.push_back(uniformRandomInt(0, numOutcomes - 1));
	}

	vector<uint32_t> counts1(numSamples, 0);
	vector<uint32_t> counts2(numSamples, 0);
	for (uint32_t i = 0; i < numSamples; ++i)
	{
		++counts1[samples1[i]];
