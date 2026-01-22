		return(a->nid);

	if (added != NULL)
		{
		ad.type=ADDED_DATA;
		ad.obj=(ASN1_OBJECT *)a; /* XXX: ugly but harmless */
		adp=(ADDED_OBJ *)lh_retrieve(added,&ad);
		if (adp != NULL) return (adp->obj->nid);
		}
	op=(ASN1_OBJECT **)OBJ_bsearch((char *)&a,(char *)obj_objs,NUM_OBJ,
		sizeof(ASN1_OBJECT *),obj_cmp);
	if (op == NULL)
		return(NID_undef);
	return((*op)->nid);
	}

/* Convert an object name into an ASN1_OBJECT
 * if "noname" is not set then search for short and long names first.
 * This will convert the "dotted" form into an object: unlike OBJ_txt2nid
 * it can be used with any objects, not just registered ones.
 */

ASN1_OBJECT *OBJ_txt2obj(const char *s, int no_name)
	{
	int nid = NID_undef;
	ASN1_OBJECT *op=NULL;
	unsigned char *buf,*p;
	int i, j;

	if(!no_name) {
		if( ((nid = OBJ_sn2nid(s)) != NID_undef) ||
			((nid = OBJ_ln2nid(s)) != NID_undef) ) 
					return OBJ_nid2obj(nid);
	}

	/* Work out size of content octets */
	i=a2d_ASN1_OBJECT(NULL,0,s,-1);
	if (i <= 0) {
		/* Clear the error */
		ERR_get_error();
		return NULL;
	}
	/* Work out total size */
	j = ASN1_object_size(0,i,V_ASN1_OBJECT);

	if((buf=(unsigned char *)OPENSSL_malloc(j)) == NULL) return NULL;

	p = buf;
	/* Write out tag+length */
	ASN1_put_object(&p,0,i,V_ASN1_OBJECT,V_ASN1_UNIVERSAL);
	/* Write out contents */
	a2d_ASN1_OBJECT(p,i,s,-1);
	
	p=buf;
	op=d2i_ASN1_OBJECT(NULL,&p,i);
	OPENSSL_free(buf);
	return op;
	}

int OBJ_obj2txt(char *buf, int buf_len, const ASN1_OBJECT *a, int no_name)
{
	int i,idx=0,n=0,len,nid;
	unsigned long l;
	unsigned char *p;
