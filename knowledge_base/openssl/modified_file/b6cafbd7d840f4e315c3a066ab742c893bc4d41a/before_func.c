 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com).
 *
 */

#include "cryptlib.h"
#include "bn_lcl.h"

static BIGNUM *euclid(BIGNUM *a, BIGNUM *b);

int BN_gcd(BIGNUM *r, const BIGNUM *in_a, const BIGNUM *in_b, BN_CTX *ctx)
	{
	BIGNUM *a,*b,*t;
	int ret=0;

	bn_check_top(in_a);
	bn_check_top(in_b);

	BN_CTX_start(ctx);
	a = BN_CTX_get(ctx);
	b = BN_CTX_get(ctx);
	if (a == NULL || b == NULL) goto err;

	if (BN_copy(a,in_a) == NULL) goto err;
	if (BN_copy(b,in_b) == NULL) goto err;
	a->neg = 0;
	b->neg = 0;

	if (BN_cmp(a,b) < 0) { t=a; a=b; b=t; }
	t=euclid(a,b);
	if (t == NULL) goto err;

	if (BN_copy(r,t) == NULL) goto err;
	ret=1;
err:
	BN_CTX_end(ctx);
	return(ret);
	}

static BIGNUM *euclid(BIGNUM *a, BIGNUM *b)
	{
	BIGNUM *t;
	int shifts=0;

	bn_check_top(a);
	bn_check_top(b);

	/* 0 <= b <= a */
	while (!BN_is_zero(b))
		{
		/* 0 < b <= a */

		if (BN_is_odd(a))
			{
			if (BN_is_odd(b))
				{
				if (!BN_sub(a,a,b)) goto err;
				if (!BN_rshift1(a,a)) goto err;
				if (BN_cmp(a,b) < 0)
					{ t=a; a=b; b=t; }
				}
			else		/* a odd - b even */
				{
				if (!BN_rshift1(b,b)) goto err;
				if (BN_cmp(a,b) < 0)
					{ t=a; a=b; b=t; }
				}
			}
		else			/* a is even */
			{
			if (BN_is_odd(b))
				{
				if (!BN_rshift1(a,a)) goto err;
				if (BN_cmp(a,b) < 0)
					{ t=a; a=b; b=t; }
				}
			else		/* a even - b even */
				{
				if (!BN_rshift1(a,a)) goto err;
				if (!BN_rshift1(b,b)) goto err;
				shifts++;
				}
			}
		/* 0 <= b <= a */
		}

	if (shifts)
		{
		if (!BN_lshift(a,a,shifts)) goto err;
		}
	return(a);
err:
	return(NULL);
	}


/* solves ax == 1 (mod n) */
BIGNUM *BN_mod_inverse(BIGNUM *in,
	const BIGNUM *a, const BIGNUM *n, BN_CTX *ctx)
	{
	BIGNUM *A,*B,*X,*Y,*M,*D,*T,*R=NULL;
	BIGNUM *ret=NULL;
	int sign;

	bn_check_top(a);
	bn_check_top(n);

	BN_CTX_start(ctx);
	A = BN_CTX_get(ctx);
	B = BN_CTX_get(ctx);
	X = BN_CTX_get(ctx);
	D = BN_CTX_get(ctx);
	M = BN_CTX_get(ctx);
	Y = BN_CTX_get(ctx);
	T = BN_CTX_get(ctx);
	if (T == NULL) goto err;

	if (in == NULL)
		R=BN_new();
	else
		R=in;
	if (R == NULL) goto err;

	BN_one(X);
	BN_zero(Y);
	if (BN_copy(B,a) == NULL) goto err;
	if (BN_copy(A,n) == NULL) goto err;
	A->neg = 0;
	if (B->neg || (BN_ucmp(B, A) >= 0))
		{
		if (!BN_nnmod(B, B, A, ctx)) goto err;
		}
	sign = -1;
	/* From  B = a mod |n|,  A = |n|  it follows that
	 *
	 *      0 <= B < A,
	 *           X*a  ==  B   (mod |n|),
	 *     -sign*Y*a  ==  A   (mod |n|).
	 */

	while (!BN_is_zero(B))
		{
		BIGNUM *tmp;

		/*
		 *      0 < B < A,
		 * (*)       X*a  ==  B   (mod |n|),
