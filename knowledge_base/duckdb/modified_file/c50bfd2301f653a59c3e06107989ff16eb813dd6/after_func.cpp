OR
TRUE  OR TRUE  = TRUE
TRUE  OR FALSE = TRUE
TRUE  OR NULL  = TRUE
FALSE OR TRUE  = TRUE
FALSE OR FALSE = FALSE
FALSE OR NULL  = NULL
NULL  OR TRUE  = TRUE
NULL  OR FALSE = NULL
NULL  OR NULL  = NULL

Basically:
- Only false if both are false
- True if either is true (regardless of NULLs)
- NULL otherwise
*/
namespace operators {
struct OrMask {
	static inline bool Operation(bool left, bool right, bool left_null,
	                             bool right_null) {
		return (left_null && (right_null || !right)) || (right_null && !left);
	}
};
} // namespace operators
void VectorOperations::Or(Vector &left, Vector &right, Vector &result) {
