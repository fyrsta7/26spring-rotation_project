			cout << "Running " << step << endl;
		allSteps().at(step)->run(m_context, _ast);
		if (m_debug == Debug::PrintChanges)
		{
			// TODO should add switch to also compare variable names!
			if (SyntacticallyEqual{}.statementEqual(_ast, *copy))
				cout << "== Running " << step << " did not cause changes." << endl;
			else
			{
				cout << "== Running " << step << " changed the AST." << endl;
				cout << AsmPrinter{}(_ast) << endl;
				copy = make_unique<Block>(std::get<Block>(ASTCopier{}(_ast)));
			}
		}
	}
}

