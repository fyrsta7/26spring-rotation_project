int OBJ_obj2txt(char *buf, int buf_len, const ASN1_OBJECT *a, int no_name)
{
	int i,idx=0,n=0,len,nid;
	unsigned long l;
	unsigned char *p;
	const char *s;
	char tbuf[32];

	if (buf_len <= 0) return(0);

	if ((a == NULL) || (a->data == NULL)) {
		buf[0]='\0';
		return(0);
	}

	if (no_name || (nid=OBJ_obj2nid(a)) == NID_undef) {
		len=a->length;
		p=a->data;

		idx=0;
		l=0;
		while (idx < a->length) {
			l|=(p[idx]&0x7f);
			if (!(p[idx] & 0x80)) break;
			l<<=7L;
			idx++;
		}
		idx++;
		i=(int)(l/40);
		if (i > 2) i=2;
		l-=(long)(i*40);

		sprintf(tbuf,"%d.%lu",i,l);
		i=strlen(tbuf);
		strncpy(buf,tbuf,buf_len);
		buf_len-=i;
		buf+=i;
		n+=i;

		l=0;
		for (; idx<len; idx++) {
			l|=p[idx]&0x7f;
			if (!(p[idx] & 0x80)) {
				sprintf(tbuf,".%lu",l);
				i=strlen(tbuf);
				if (buf_len > 0)
					strncpy(buf,tbuf,buf_len);
				buf_len-=i;
				buf+=i;
				n+=i;
				l=0;
			}
			l<<=7L;
		}
	} else {
		s=OBJ_nid2ln(nid);
		if (s == NULL)
			s=OBJ_nid2sn(nid);
		strncpy(buf,s,buf_len);
		n=strlen(s);
	}
	buf[buf_len-1]='\0';
	return(n);
}