#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

namespace {

// Find root node of the colocation group.
// The map is mapping from one node name to its parent. node_name is the
// starting node to search. By iteratively following the path from child to
// parent, we can find the root node for the colocation group that node_name
// belongs to.
string GetColocationGroupRoot(std::unordered_map<string, string>* map,
                              const string& node_name) {
