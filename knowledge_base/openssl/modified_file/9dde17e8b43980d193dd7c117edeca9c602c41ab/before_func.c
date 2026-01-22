	if (a != r)
		{
		if (bn_wexpand(r,a->top) == NULL) return(0);
		r->top=a->top;
		r->neg=a->neg;
		}
	ap=a->d;
	rp=r->d;
	c=0;
	for (i=a->top-1; i>=0; i--)
		{
		t=ap[i];
		rp[i]=((t>>1)&BN_MASK2)|c;
		c=(t&1)?BN_TBIT:0;
		}
	bn_correct_top(r);
	bn_check_top(r);
	return(1);
	}

int BN_lshift(BIGNUM *r, const BIGNUM *a, int n)
	{
	int i,nw,lb,rb;
	BN_ULONG *t,*f;
	BN_ULONG l;

	bn_check_top(r);
	bn_check_top(a);

	r->neg=a->neg;
	nw=n/BN_BITS2;
	if (bn_wexpand(r,a->top+nw+1) == NULL) return(0);
	lb=n%BN_BITS2;
	rb=BN_BITS2-lb;
	f=a->d;
	t=r->d;
	t[a->top+nw]=0;
	if (lb == 0)
		for (i=a->top-1; i>=0; i--)
			t[nw+i]=f[i];
	else
		for (i=a->top-1; i>=0; i--)
			{
			l=f[i];
			t[nw+i+1]|=(l>>rb)&BN_MASK2;
			t[nw+i]=(l<<lb)&BN_MASK2;
			}
	memset(t,0,nw*sizeof(t[0]));
/*	for (i=0; i<nw; i++)
		t[i]=0;*/
	r->top=a->top+nw+1;
	bn_correct_top(r);
	bn_check_top(r);
	return(1);
