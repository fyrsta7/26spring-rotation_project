static void extractFunctions(ASTPtr expression, const NameSet & columns, std::vector<ASTPtr> & result)
{
	if (const ASTFunction * function = typeid_cast<const ASTFunction *>(&* expression))
	{
		if (function->name == "and")
		{
			for (size_t i = 0; i < function->arguments->children.size(); ++i)
				extractFunctions(function->arguments->children[i], columns, result);
		}
		else
		{
			if (isValidFunction(expression, columns))
				result.push_back(expression->clone());
		}
	}
}
