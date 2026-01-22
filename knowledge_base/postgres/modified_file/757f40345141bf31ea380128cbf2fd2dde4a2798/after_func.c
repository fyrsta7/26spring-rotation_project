		return true;
	else
		return false;
}

/*
 * equal_indexkey_var--
 *	  Returns t iff an index key 'index-key' matches the corresponding
 *	  fields of var node 'var'.
 *
 */
static bool
equal_indexkey_var(int index_key, Var *var)
{
	if (index_key == var->varattno)
		return true;
	else
