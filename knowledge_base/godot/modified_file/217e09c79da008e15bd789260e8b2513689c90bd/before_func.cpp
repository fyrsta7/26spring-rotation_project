
				} break;
			}

			int dst_addr=(p_stack_level)|(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
			codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
			codegen.alloc_stack(p_stack_level);
			return dst_addr;
		} break;
		//TYPE_TYPE,
		default: {

			ERR_EXPLAIN("Bug in bytecode compiler, unexpected node in parse tree while parsing expression.");
			ERR_FAIL_V(-1); //unreachable code
		} break;


	}

	ERR_FAIL_V(-1); //unreachable code
}


Error GDCompiler::_parse_block(CodeGen& codegen,const GDParser::BlockNode *p_block,int p_stack_level,int p_break_addr,int p_continue_addr) {

	codegen.push_stack_identifiers();
	int new_identifiers=0;
	codegen.current_line=p_block->line;

	for(int i=0;i<p_block->statements.size();i++) {

		const GDParser::Node *s = p_block->statements[i];


		switch(s->type) {
			case GDParser::Node::TYPE_NEWLINE: {

				const GDParser::NewLineNode *nl = static_cast<const GDParser::NewLineNode*>(s);
				codegen.opcodes.push_back(GDFunction::OPCODE_LINE);
				codegen.opcodes.push_back(nl->line);
				codegen.current_line=nl->line;

			} break;
			case GDParser::Node::TYPE_CONTROL_FLOW: {
				// try subblocks

				const GDParser::ControlFlowNode *cf = static_cast<const GDParser::ControlFlowNode*>(s);

				switch(cf->cf_type) {


					case GDParser::ControlFlowNode::CF_IF: {

#ifdef DEBUG_ENABLED
						codegen.opcodes.push_back(GDFunction::OPCODE_LINE);
						codegen.opcodes.push_back(cf->line);
						codegen.current_line=cf->line;
#endif
						int ret = _parse_expression(codegen,cf->arguments[0],p_stack_level,false);
						if (ret<0)
							return ERR_PARSE_ERROR;

						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP_IF_NOT);
						codegen.opcodes.push_back(ret);
						int else_addr=codegen.opcodes.size();
						codegen.opcodes.push_back(0); //temporary

						Error err = _parse_block(codegen,cf->body,p_stack_level,p_break_addr,p_continue_addr);
						if (err)
							return err;

						if (cf->body_else) {

							codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
							int end_addr=codegen.opcodes.size();
							codegen.opcodes.push_back(0);
							codegen.opcodes[else_addr]=codegen.opcodes.size();

							Error err = _parse_block(codegen,cf->body_else,p_stack_level,p_break_addr,p_continue_addr);
							if (err)
								return err;

							codegen.opcodes[end_addr]=codegen.opcodes.size();
						} else {
							//end without else
							codegen.opcodes[else_addr]=codegen.opcodes.size();

						}

					} break;
					case GDParser::ControlFlowNode::CF_FOR: {



						int slevel=p_stack_level;
						int iter_stack_pos=slevel;
						int iterator_pos = (slevel++)|(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
						int counter_pos = (slevel++)|(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
						int container_pos = (slevel++)|(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
						codegen.alloc_stack(slevel);

						    codegen.push_stack_identifiers();
						      codegen.add_stack_identifier(static_cast<const GDParser::IdentifierNode*>(cf->arguments[0])->name,iter_stack_pos);

						int ret = _parse_expression(codegen,cf->arguments[1],slevel,false);
						if (ret<0)
							return ERR_COMPILATION_FAILED;

						//assign container
						codegen.opcodes.push_back(GDFunction::OPCODE_ASSIGN);
						codegen.opcodes.push_back(container_pos);
						codegen.opcodes.push_back(ret);

						//begin loop
						codegen.opcodes.push_back(GDFunction::OPCODE_ITERATE_BEGIN);
						codegen.opcodes.push_back(counter_pos);
						codegen.opcodes.push_back(container_pos);
						codegen.opcodes.push_back(codegen.opcodes.size()+4);
						codegen.opcodes.push_back(iterator_pos);
						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP); //skip code for next
						codegen.opcodes.push_back(codegen.opcodes.size()+8);
						//break loop
						int break_pos=codegen.opcodes.size();
						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP); //skip code for next
						codegen.opcodes.push_back(0); //skip code for next
						//next loop
						int continue_pos=codegen.opcodes.size();
						codegen.opcodes.push_back(GDFunction::OPCODE_ITERATE);
						codegen.opcodes.push_back(counter_pos);
						codegen.opcodes.push_back(container_pos);
						codegen.opcodes.push_back(break_pos);
						codegen.opcodes.push_back(iterator_pos);


						Error err = _parse_block(codegen,cf->body,slevel,break_pos,continue_pos);
						if (err)
							return err;


						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
						codegen.opcodes.push_back(continue_pos);
						codegen.opcodes[break_pos+1]=codegen.opcodes.size();


						codegen.pop_stack_identifiers();

					} break;
					case GDParser::ControlFlowNode::CF_WHILE: {

						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
						codegen.opcodes.push_back(codegen.opcodes.size()+3);
						int break_addr=codegen.opcodes.size();
						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
						codegen.opcodes.push_back(0);
						int continue_addr=codegen.opcodes.size();

						int ret = _parse_expression(codegen,cf->arguments[0],p_stack_level,false);
						if (ret<0)
							return ERR_PARSE_ERROR;
						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP_IF_NOT);
						codegen.opcodes.push_back(ret);
						codegen.opcodes.push_back(break_addr);
						Error err = _parse_block(codegen,cf->body,p_stack_level,break_addr,continue_addr);
						if (err)
							return err;
						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
						codegen.opcodes.push_back(continue_addr);

						codegen.opcodes[break_addr+1]=codegen.opcodes.size();

					} break;
					case GDParser::ControlFlowNode::CF_SWITCH: {

					} break;
					case GDParser::ControlFlowNode::CF_BREAK: {

						if (p_break_addr<0) {

							_set_error("'break'' not within loop",cf);
							return ERR_COMPILATION_FAILED;
						}
						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
						codegen.opcodes.push_back(p_break_addr);

					} break;
					case GDParser::ControlFlowNode::CF_CONTINUE: {

						if (p_continue_addr<0) {

							_set_error("'continue' not within loop",cf);
							return ERR_COMPILATION_FAILED;
						}

						codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
						codegen.opcodes.push_back(p_continue_addr);

					} break;
					case GDParser::ControlFlowNode::CF_RETURN: {

						int ret;

						if (cf->arguments.size()) {

							ret = _parse_expression(codegen,cf->arguments[0],p_stack_level,false);
							if (ret<0)
								return ERR_PARSE_ERROR;

						} else {

							ret=GDFunction::ADDR_TYPE_NIL << GDFunction::ADDR_BITS;
						}

						codegen.opcodes.push_back(GDFunction::OPCODE_RETURN);
						codegen.opcodes.push_back(ret);

					} break;

				}
			} break;
			case GDParser::Node::TYPE_ASSERT: {
				// try subblocks

				const GDParser::AssertNode *as = static_cast<const GDParser::AssertNode*>(s);

				int ret = _parse_expression(codegen,as->condition,p_stack_level,false);
				if (ret<0)
					return ERR_PARSE_ERROR;

				codegen.opcodes.push_back(GDFunction::OPCODE_ASSERT);
				codegen.opcodes.push_back(ret);
			} break;
			case GDParser::Node::TYPE_BREAKPOINT: {
				// try subblocks
				codegen.opcodes.push_back(GDFunction::OPCODE_BREAKPOINT);
			} break;
