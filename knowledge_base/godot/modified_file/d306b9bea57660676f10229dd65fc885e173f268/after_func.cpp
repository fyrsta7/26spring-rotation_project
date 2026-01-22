
	if (var_op==Variant::OP_MAX) {

		return _parse_expression(codegen,p_expression->arguments[1],p_stack_level,false,initializer);
	}

	if (!_create_binary_operator(codegen,p_expression,var_op,p_stack_level,initializer))
		return -1;

	int dst_addr=(p_stack_level)|(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
	codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
	codegen.alloc_stack(p_stack_level);
	return dst_addr;

}

int GDCompiler::_parse_expression(CodeGen& codegen,const GDParser::Node *p_expression, int p_stack_level,bool p_root,bool p_initializer) {


	switch(p_expression->type) {
		//should parse variable declaration and adjust stack accordingly...
		case GDParser::Node::TYPE_IDENTIFIER: {
			//return identifier
			//wait, identifier could be a local variable or something else... careful here, must reference properly
			//as stack may be more interesting to work with

			//This could be made much simpler by just indexing "self", but done this way (with custom self-addressing modes) increases peformance a lot.

			const GDParser::IdentifierNode *in = static_cast<const GDParser::IdentifierNode*>(p_expression);

			StringName identifier = in->name;

			// TRY STACK!
			if (!p_initializer && codegen.stack_identifiers.has(identifier)) {

				int pos = codegen.stack_identifiers[identifier];
				return pos|(GDFunction::ADDR_TYPE_STACK_VARIABLE<<GDFunction::ADDR_BITS);

			}
			//TRY MEMBERS!
			if (!codegen.function_node || !codegen.function_node->_static) {

				// TRY MEMBER VARIABLES!
				//static function
				if (codegen.script->member_indices.has(identifier)) {

					int idx = codegen.script->member_indices[identifier].index;
					return idx|(GDFunction::ADDR_TYPE_MEMBER<<GDFunction::ADDR_BITS); //argument (stack root)
				}
			}

			//TRY CLASS CONSTANTS

			GDScript *owner = codegen.script;
			while (owner) {

				GDScript *scr = owner;
				GDNativeClass *nc=NULL;
				while(scr) {

					if (scr->constants.has(identifier)) {

						//int idx=scr->constants[identifier];
						int idx = codegen.get_name_map_pos(identifier);
						return idx|(GDFunction::ADDR_TYPE_CLASS_CONSTANT<<GDFunction::ADDR_BITS); //argument (stack root)
					}
					if (scr->native.is_valid())
						nc=scr->native.ptr();
					scr=scr->_base;
				}

				// CLASS C++ Integer Constant

				if (nc) {

					bool success=false;
					int constant = ObjectTypeDB::get_integer_constant(nc->get_name(),identifier,&success);
					if (success) {
						Variant key=constant;
						int idx;

						if (!codegen.constant_map.has(key)) {

							idx=codegen.constant_map.size();
							codegen.constant_map[key]=idx;

						} else {
							idx=codegen.constant_map[key];
						}

						return idx|(GDFunction::ADDR_TYPE_LOCAL_CONSTANT<<GDFunction::ADDR_BITS); //make it a local constant (faster access)
					}

				}

				owner=owner->_owner;
			}

			/*
			 handled in constants now
			 if (codegen.script->subclasses.has(identifier)) {
				//same with a subclass, make it a local constant.
				int idx = codegen.get_constant_pos(codegen.script->subclasses[identifier]);
				return idx|(GDFunction::ADDR_TYPE_LOCAL_CONSTANT<<GDFunction::ADDR_BITS); //make it a local constant (faster access)

			}*/

			if (GDScriptLanguage::get_singleton()->get_global_map().has(identifier)) {

				int idx = GDScriptLanguage::get_singleton()->get_global_map()[identifier];
				return idx|(GDFunction::ADDR_TYPE_GLOBAL<<GDFunction::ADDR_BITS); //argument (stack root)
			}

			//not found, error

			_set_error("Identifier not found: "+String(identifier),p_expression);

			return -1;


		} break;
		case GDParser::Node::TYPE_CONSTANT: {
			//return constant
			const GDParser::ConstantNode *cn = static_cast<const GDParser::ConstantNode*>(p_expression);


			int idx;

			if (!codegen.constant_map.has(cn->value)) {

				idx=codegen.constant_map.size();
				codegen.constant_map[cn->value]=idx;

			} else {
				idx=codegen.constant_map[cn->value];
			}


			return idx|(GDFunction::ADDR_TYPE_LOCAL_CONSTANT<<GDFunction::ADDR_BITS); //argument (stack root)

		} break;
		case GDParser::Node::TYPE_SELF: {
			//return constant
			if (codegen.function_node && codegen.function_node->_static) {
				_set_error("'self' not present in static function!",p_expression);
				return -1;
			}
			return (GDFunction::ADDR_TYPE_SELF<<GDFunction::ADDR_BITS);
		} break;
		case GDParser::Node::TYPE_ARRAY: {

			const GDParser::ArrayNode *an = static_cast<const GDParser::ArrayNode*>(p_expression);
			Vector<int> values;

			int slevel=p_stack_level;

			for(int i=0;i<an->elements.size();i++) {

				int ret = _parse_expression(codegen,an->elements[i],slevel);
				if (ret<0)
					return ret;
				if (ret&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
					slevel++;
					codegen.alloc_stack(slevel);
				}

				values.push_back(ret);
			}

			codegen.opcodes.push_back(GDFunction::OPCODE_CONSTRUCT_ARRAY);
			codegen.opcodes.push_back(values.size());
			for(int i=0;i<values.size();i++)
				codegen.opcodes.push_back(values[i]);

			int dst_addr=(p_stack_level)|(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
			codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
			codegen.alloc_stack(p_stack_level);
			return dst_addr;

		} break;
		case GDParser::Node::TYPE_DICTIONARY: {

			const GDParser::DictionaryNode *dn = static_cast<const GDParser::DictionaryNode*>(p_expression);
			Vector<int> values;

			int slevel=p_stack_level;

			for(int i=0;i<dn->elements.size();i++) {

				int ret = _parse_expression(codegen,dn->elements[i].key,slevel);
				if (ret<0)
					return ret;
				if (ret&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
					slevel++;
					codegen.alloc_stack(slevel);
				}

				values.push_back(ret);

				ret = _parse_expression(codegen,dn->elements[i].value,slevel);
				if (ret<0)
					return ret;
				if (ret&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
					slevel++;
					codegen.alloc_stack(slevel);
				}

				values.push_back(ret);
			}

			codegen.opcodes.push_back(GDFunction::OPCODE_CONSTRUCT_DICTIONARY);
			codegen.opcodes.push_back(dn->elements.size());
			for(int i=0;i<values.size();i++)
				codegen.opcodes.push_back(values[i]);

			int dst_addr=(p_stack_level)|(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
			codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
			codegen.alloc_stack(p_stack_level);
			return dst_addr;

		} break;
		case GDParser::Node::TYPE_OPERATOR: {
			//hell breaks loose

			const GDParser::OperatorNode *on = static_cast<const GDParser::OperatorNode*>(p_expression);
			switch(on->op) {


				//call/constructor operator
				case GDParser::OperatorNode::OP_PARENT_CALL: {


					ERR_FAIL_COND_V(on->arguments.size()<1,-1);

					const GDParser::IdentifierNode *in = (const GDParser::IdentifierNode *)on->arguments[0];


					Vector<int> arguments;
					int slevel = p_stack_level;
					for(int i=1;i<on->arguments.size();i++) {

						int ret = _parse_expression(codegen,on->arguments[i],slevel);
						if (ret<0)
							return ret;
						if (ret&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
							slevel++;
							codegen.alloc_stack(slevel);
						}
						arguments.push_back(ret);
					}

					//push call bytecode
					codegen.opcodes.push_back(GDFunction::OPCODE_CALL_SELF_BASE); // basic type constructor

					codegen.opcodes.push_back(codegen.get_name_map_pos(in->name)); //instance
					codegen.opcodes.push_back(arguments.size()); //argument count
					codegen.alloc_call(arguments.size());
					for(int i=0;i<arguments.size();i++)
						codegen.opcodes.push_back(arguments[i]); //arguments

				} break;
				case GDParser::OperatorNode::OP_CALL: {

					if (on->arguments[0]->type==GDParser::Node::TYPE_TYPE) {
						//construct a basic type
						ERR_FAIL_COND_V(on->arguments.size()<1,-1);

						const GDParser::TypeNode *tn = (const GDParser::TypeNode *)on->arguments[0];
						int vtype = tn->vtype;

						Vector<int> arguments;
						int slevel = p_stack_level;
						for(int i=1;i<on->arguments.size();i++) {

							int ret = _parse_expression(codegen,on->arguments[i],slevel);
							if (ret<0)
								return ret;
							if (ret&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
								slevel++;
								codegen.alloc_stack(slevel);
							}
							arguments.push_back(ret);
						}

						//push call bytecode
						codegen.opcodes.push_back(GDFunction::OPCODE_CONSTRUCT); // basic type constructor
						codegen.opcodes.push_back(vtype); //instance
						codegen.opcodes.push_back(arguments.size()); //argument count
						codegen.alloc_call(arguments.size());
						for(int i=0;i<arguments.size();i++)
							codegen.opcodes.push_back(arguments[i]); //arguments

					} else if (on->arguments[0]->type==GDParser::Node::TYPE_BUILT_IN_FUNCTION) {
						//built in function

						ERR_FAIL_COND_V(on->arguments.size()<1,-1);


						Vector<int> arguments;
						int slevel = p_stack_level;
						for(int i=1;i<on->arguments.size();i++) {

							int ret = _parse_expression(codegen,on->arguments[i],slevel);
							if (ret<0)
								return ret;

							if (ret&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
								slevel++;
								codegen.alloc_stack(slevel);
							}

							arguments.push_back(ret);
						}


						codegen.opcodes.push_back(GDFunction::OPCODE_CALL_BUILT_IN);
						codegen.opcodes.push_back(static_cast<const GDParser::BuiltInFunctionNode*>(on->arguments[0])->function);
						codegen.opcodes.push_back(on->arguments.size()-1);
						codegen.alloc_call(on->arguments.size()-1);
						for(int i=0;i<arguments.size();i++)
							codegen.opcodes.push_back(arguments[i]);

					} else {
						//regular function
						ERR_FAIL_COND_V(on->arguments.size()<2,-1);

						const GDParser::Node *instance = on->arguments[0];

						bool in_static=false;
						if (instance->type==GDParser::Node::TYPE_SELF) {
							//room for optimization

						}


						Vector<int> arguments;
						int slevel = p_stack_level;

						for(int i=0;i<on->arguments.size();i++) {

							int ret;

							if (i==0 && on->arguments[i]->type==GDParser::Node::TYPE_SELF && codegen.function_node && codegen.function_node->_static) {
								//static call to self
								ret=(GDFunction::ADDR_TYPE_CLASS<<GDFunction::ADDR_BITS);
							} else if (i==1) {

								if (on->arguments[i]->type!=GDParser::Node::TYPE_IDENTIFIER) {
									_set_error("Attempt to call a non-identifier.",on);
									return -1;
								}
								GDParser::IdentifierNode *id = static_cast<GDParser::IdentifierNode*>(on->arguments[i]);
								ret=codegen.get_name_map_pos(id->name);

							} else {

								ret = _parse_expression(codegen,on->arguments[i],slevel);
								if (ret<0)
									return ret;
								if (ret&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
									slevel++;
									codegen.alloc_stack(slevel);
								}
							}
							arguments.push_back(ret);

						}

						codegen.opcodes.push_back(p_root?GDFunction::OPCODE_CALL:GDFunction::OPCODE_CALL_RETURN); // perform operator
						codegen.opcodes.push_back(on->arguments.size()-2);
						codegen.alloc_call(on->arguments.size()-2);
						for(int i=0;i<arguments.size();i++)
							codegen.opcodes.push_back(arguments[i]);
					}
				} break;
				case GDParser::OperatorNode::OP_YIELD: {


					ERR_FAIL_COND_V(on->arguments.size() && on->arguments.size()!=2,-1);

					Vector<int> arguments;
					int slevel = p_stack_level;
					for(int i=0;i<on->arguments.size();i++) {

						int ret = _parse_expression(codegen,on->arguments[i],slevel);
						if (ret<0)
							return ret;
						if (ret&(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS)) {
							slevel++;
							codegen.alloc_stack(slevel);
						}
						arguments.push_back(ret);
					}

					//push call bytecode
					codegen.opcodes.push_back(arguments.size()==0?GDFunction::OPCODE_YIELD:GDFunction::OPCODE_YIELD_SIGNAL); // basic type constructor
					for(int i=0;i<arguments.size();i++)
						codegen.opcodes.push_back(arguments[i]); //arguments
					codegen.opcodes.push_back(GDFunction::OPCODE_YIELD_RESUME);
					//next will be where to place the result :)

				} break;

				//indexing operator
				case GDParser::OperatorNode::OP_INDEX:
				case GDParser::OperatorNode::OP_INDEX_NAMED: {

					ERR_FAIL_COND_V(on->arguments.size()!=2,-1);

					int slevel = p_stack_level;
					bool named=(on->op==GDParser::OperatorNode::OP_INDEX_NAMED);

					int from = _parse_expression(codegen,on->arguments[0],slevel);
					if (from<0)
						return from;

					int index;
					if (named) {
						if (on->arguments[0]->type==GDParser::Node::TYPE_SELF && codegen.script && codegen.function_node && !codegen.function_node->_static) {

							GDParser::IdentifierNode* identifier = static_cast<GDParser::IdentifierNode*>(on->arguments[1]);
							const Map<StringName,GDScript::MemberInfo>::Element *MI = codegen.script->member_indices.find(identifier->name);

#ifdef DEBUG_ENABLED
							if (MI && MI->get().getter==codegen.function_node->name) {
								String n = static_cast<GDParser::IdentifierNode*>(on->arguments[1])->name;
								_set_error("Must use '"+n+"' instead of 'self."+n+"' in getter.",on);
								return -1;
							}
#endif

							if (MI && MI->get().getter=="") {
								// Faster than indexing self (as if no self. had been used)
								return (MI->get().index)|(GDFunction::ADDR_TYPE_MEMBER<<GDFunction::ADDR_BITS);
							}
						}

						index=codegen.get_name_map_pos(static_cast<GDParser::IdentifierNode*>(on->arguments[1])->name);

					} else {

						if (on->arguments[1]->type==GDParser::Node::TYPE_CONSTANT && static_cast<const GDParser::ConstantNode*>(on->arguments[1])->value.get_type()==Variant::STRING) {
							//also, somehow, named (speed up anyway)
							StringName name = static_cast<const GDParser::ConstantNode*>(on->arguments[1])->value;
							index=codegen.get_name_map_pos(name);
							named=true;

						} else {
							//regular indexing
							if (from&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
								slevel++;
								codegen.alloc_stack(slevel);
							}

							index = _parse_expression(codegen,on->arguments[1],slevel);
							if (index<0)
								return index;
						}
					}

					codegen.opcodes.push_back(named?GDFunction::OPCODE_GET_NAMED:GDFunction::OPCODE_GET); // perform operator
					codegen.opcodes.push_back(from); // argument 1
					codegen.opcodes.push_back(index); // argument 2 (unary only takes one parameter)

				} break;
				case GDParser::OperatorNode::OP_AND: {

					// AND operator with early out on failure

					int res = _parse_expression(codegen,on->arguments[0],p_stack_level);
					if (res<0)
						return res;
					codegen.opcodes.push_back(GDFunction::OPCODE_JUMP_IF_NOT);
					codegen.opcodes.push_back(res);
					int jump_fail_pos=codegen.opcodes.size();
					codegen.opcodes.push_back(0);

					res = _parse_expression(codegen,on->arguments[1],p_stack_level);
					if (res<0)
						return res;

					codegen.opcodes.push_back(GDFunction::OPCODE_JUMP_IF_NOT);
					codegen.opcodes.push_back(res);
					int jump_fail_pos2=codegen.opcodes.size();
					codegen.opcodes.push_back(0);

					codegen.alloc_stack(p_stack_level); //it will be used..
					codegen.opcodes.push_back(GDFunction::OPCODE_ASSIGN_TRUE);
					codegen.opcodes.push_back(p_stack_level|GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
					codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
					codegen.opcodes.push_back(codegen.opcodes.size()+3);
					codegen.opcodes[jump_fail_pos]=codegen.opcodes.size();
					codegen.opcodes[jump_fail_pos2]=codegen.opcodes.size();
					codegen.opcodes.push_back(GDFunction::OPCODE_ASSIGN_FALSE);
					codegen.opcodes.push_back(p_stack_level|GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
					return p_stack_level|GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS;

				} break;
				case GDParser::OperatorNode::OP_OR: {

					// OR operator with early out on success

					int res = _parse_expression(codegen,on->arguments[0],p_stack_level);
					if (res<0)
						return res;
					codegen.opcodes.push_back(GDFunction::OPCODE_JUMP_IF);
					codegen.opcodes.push_back(res);
					int jump_success_pos=codegen.opcodes.size();
					codegen.opcodes.push_back(0);

					res = _parse_expression(codegen,on->arguments[1],p_stack_level);
					if (res<0)
						return res;

					codegen.opcodes.push_back(GDFunction::OPCODE_JUMP_IF);
					codegen.opcodes.push_back(res);
					int jump_success_pos2=codegen.opcodes.size();
					codegen.opcodes.push_back(0);

					codegen.alloc_stack(p_stack_level); //it will be used..
					codegen.opcodes.push_back(GDFunction::OPCODE_ASSIGN_FALSE);
					codegen.opcodes.push_back(p_stack_level|GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
					codegen.opcodes.push_back(GDFunction::OPCODE_JUMP);
					codegen.opcodes.push_back(codegen.opcodes.size()+3);
					codegen.opcodes[jump_success_pos]=codegen.opcodes.size();
					codegen.opcodes[jump_success_pos2]=codegen.opcodes.size();
					codegen.opcodes.push_back(GDFunction::OPCODE_ASSIGN_TRUE);
					codegen.opcodes.push_back(p_stack_level|GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS);
					return p_stack_level|GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS;

				} break;
				//unary operators
				case GDParser::OperatorNode::OP_NEG: { if (!_create_unary_operator(codegen,on,Variant::OP_NEGATE,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_NOT: { if (!_create_unary_operator(codegen,on,Variant::OP_NOT,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_BIT_INVERT: { if (!_create_unary_operator(codegen,on,Variant::OP_BIT_NEGATE,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_PREINC: { } break; //?
				case GDParser::OperatorNode::OP_PREDEC: { } break;
				case GDParser::OperatorNode::OP_INC: { } break;
				case GDParser::OperatorNode::OP_DEC: { } break;
				//binary operators (in precedence order)
				case GDParser::OperatorNode::OP_IN: { if (!_create_binary_operator(codegen,on,Variant::OP_IN,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_EQUAL: { if (!_create_binary_operator(codegen,on,Variant::OP_EQUAL,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_NOT_EQUAL: { if (!_create_binary_operator(codegen,on,Variant::OP_NOT_EQUAL,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_LESS: { if (!_create_binary_operator(codegen,on,Variant::OP_LESS,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_LESS_EQUAL: { if (!_create_binary_operator(codegen,on,Variant::OP_LESS_EQUAL,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_GREATER: { if (!_create_binary_operator(codegen,on,Variant::OP_GREATER,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_GREATER_EQUAL: { if (!_create_binary_operator(codegen,on,Variant::OP_GREATER_EQUAL,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_ADD: { if (!_create_binary_operator(codegen,on,Variant::OP_ADD,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_SUB: { if (!_create_binary_operator(codegen,on,Variant::OP_SUBSTRACT,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_MUL: { if (!_create_binary_operator(codegen,on,Variant::OP_MULTIPLY,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_DIV: { if (!_create_binary_operator(codegen,on,Variant::OP_DIVIDE,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_MOD: { if (!_create_binary_operator(codegen,on,Variant::OP_MODULE,p_stack_level)) return -1;} break;
				//case GDParser::OperatorNode::OP_SHIFT_LEFT: { if (!_create_binary_operator(codegen,on,Variant::OP_SHIFT_LEFT,p_stack_level)) return -1;} break;
				//case GDParser::OperatorNode::OP_SHIFT_RIGHT: { if (!_create_binary_operator(codegen,on,Variant::OP_SHIFT_RIGHT,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_BIT_AND: { if (!_create_binary_operator(codegen,on,Variant::OP_BIT_AND,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_BIT_OR: { if (!_create_binary_operator(codegen,on,Variant::OP_BIT_OR,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_BIT_XOR: { if (!_create_binary_operator(codegen,on,Variant::OP_BIT_XOR,p_stack_level)) return -1;} break;
                //shift
				case GDParser::OperatorNode::OP_SHIFT_LEFT: { if (!_create_binary_operator(codegen,on,Variant::OP_SHIFT_LEFT,p_stack_level)) return -1;} break;
				case GDParser::OperatorNode::OP_SHIFT_RIGHT: { if (!_create_binary_operator(codegen,on,Variant::OP_SHIFT_RIGHT,p_stack_level)) return -1;} break;
				//assignment operators
				case GDParser::OperatorNode::OP_ASSIGN_ADD:
				case GDParser::OperatorNode::OP_ASSIGN_SUB:
				case GDParser::OperatorNode::OP_ASSIGN_MUL:
				case GDParser::OperatorNode::OP_ASSIGN_DIV:
				case GDParser::OperatorNode::OP_ASSIGN_MOD:
				case GDParser::OperatorNode::OP_ASSIGN_SHIFT_LEFT:
				case GDParser::OperatorNode::OP_ASSIGN_SHIFT_RIGHT:
				case GDParser::OperatorNode::OP_ASSIGN_BIT_AND:
				case GDParser::OperatorNode::OP_ASSIGN_BIT_OR:
				case GDParser::OperatorNode::OP_ASSIGN_BIT_XOR:
				case GDParser::OperatorNode::OP_INIT_ASSIGN:
				case GDParser::OperatorNode::OP_ASSIGN: {

					ERR_FAIL_COND_V(on->arguments.size()!=2,-1);


					if (on->arguments[0]->type==GDParser::Node::TYPE_OPERATOR && (static_cast<GDParser::OperatorNode*>(on->arguments[0])->op==GDParser::OperatorNode::OP_INDEX || static_cast<GDParser::OperatorNode*>(on->arguments[0])->op==GDParser::OperatorNode::OP_INDEX_NAMED)) {
						//SET (chained) MODE!!


#ifdef DEBUG_ENABLED
						if (static_cast<GDParser::OperatorNode*>(on->arguments[0])->op==GDParser::OperatorNode::OP_INDEX_NAMED) {
							const GDParser::OperatorNode* inon = static_cast<GDParser::OperatorNode*>(on->arguments[0]);


							if (inon->arguments[0]->type==GDParser::Node::TYPE_SELF && codegen.script && codegen.function_node && !codegen.function_node->_static) {

								const Map<StringName,GDScript::MemberInfo>::Element *MI = codegen.script->member_indices.find(static_cast<GDParser::IdentifierNode*>(inon->arguments[1])->name);
								if (MI && MI->get().setter==codegen.function_node->name) {
									String n = static_cast<GDParser::IdentifierNode*>(inon->arguments[1])->name;
									_set_error("Must use '"+n+"' instead of 'self."+n+"' in setter.",inon);
									return -1;
								}
							}
						}
#endif


						int slevel=p_stack_level;

						GDParser::OperatorNode* op = static_cast<GDParser::OperatorNode*>(on->arguments[0]);

						/* Find chain of sets */

						List<GDParser::OperatorNode*> chain;

						{
							//create get/set chain
							GDParser::OperatorNode* n=op;
							while(true) {

								chain.push_back(n);
								if (n->arguments[0]->type!=GDParser::Node::TYPE_OPERATOR)
									break;
								n = static_cast<GDParser::OperatorNode*>(n->arguments[0]);
								if (n->op!=GDParser::OperatorNode::OP_INDEX && n->op!=GDParser::OperatorNode::OP_INDEX_NAMED)
									break;
							}
						}

						/* Chain of gets */

						//get at (potential) root stack pos, so it can be returned
						int prev_pos = _parse_expression(codegen,chain.back()->get()->arguments[0],slevel);
						if (prev_pos<0)
							return prev_pos;
						int retval=prev_pos;

						//print_line("retval: "+itos(retval));

						if (retval&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
							slevel++;
							codegen.alloc_stack(slevel);
						}


						Vector<int> setchain;

						int prev_key_idx=-1;

						for(List<GDParser::OperatorNode*>::Element *E=chain.back();E;E=E->prev()) {


							if (E==chain.front()) //ignore first
								break;

							bool named = E->get()->op==GDParser::OperatorNode::OP_INDEX_NAMED;
							int key_idx;

							if (named) {

								key_idx = codegen.get_name_map_pos(static_cast<const GDParser::IdentifierNode*>(E->get()->arguments[1])->name);
								//printf("named key %x\n",key_idx);

							} else {

								if (prev_pos&(GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS)) {
									slevel++;
									codegen.alloc_stack(slevel);
								}

								GDParser::Node *key = E->get()->arguments[1];
								key_idx = _parse_expression(codegen,key,slevel);
								//printf("expr key %x\n",key_idx);


								//stack was raised here if retval was stack but..

							}

							if (key_idx<0)
								return key_idx;

							codegen.opcodes.push_back(named ? GDFunction::OPCODE_GET_NAMED : GDFunction::OPCODE_GET);
							codegen.opcodes.push_back(prev_pos);
							codegen.opcodes.push_back(key_idx);
							slevel++;
							codegen.alloc_stack(slevel);
							int dst_pos = (GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS)|slevel;

							codegen.opcodes.push_back(dst_pos);

							//add in reverse order, since it will be reverted
							setchain.push_back(dst_pos);
							setchain.push_back(key_idx);
							setchain.push_back(prev_pos);
							setchain.push_back(named ? GDFunction::OPCODE_SET_NAMED : GDFunction::OPCODE_SET);

							prev_pos=dst_pos;
							prev_key_idx=key_idx;

						}

						setchain.invert();


						int set_index;
						bool named=false;


						if (static_cast<const GDParser::OperatorNode*>(op)->op==GDParser::OperatorNode::OP_INDEX_NAMED) {


							set_index=codegen.get_name_map_pos(static_cast<const GDParser::IdentifierNode*>(op->arguments[1])->name);
							named=true;
						} else {

							set_index = _parse_expression(codegen,op->arguments[1],slevel+1);
							named=false;
						}


						if (set_index<0)
							return set_index;

						if (set_index&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
							slevel++;
							codegen.alloc_stack(slevel);
						}


						int set_value = _parse_assign_right_expression(codegen,on,slevel+1);
						if (set_value<0)
							return set_value;

						codegen.opcodes.push_back(named?GDFunction::OPCODE_SET_NAMED:GDFunction::OPCODE_SET);
						codegen.opcodes.push_back(prev_pos);
						codegen.opcodes.push_back(set_index);
						codegen.opcodes.push_back(set_value);

						for(int i=0;i<setchain.size();i+=4) {


							codegen.opcodes.push_back(setchain[i+0]);
							codegen.opcodes.push_back(setchain[i+1]);
							codegen.opcodes.push_back(setchain[i+2]);
							codegen.opcodes.push_back(setchain[i+3]);
						}

						return retval;


					} else {
						//ASSIGNMENT MODE!!

						int slevel = p_stack_level;

						int dst_address_a = _parse_expression(codegen,on->arguments[0],slevel,false,on->op==GDParser::OperatorNode::OP_INIT_ASSIGN);
						if (dst_address_a<0)
							return -1;

						if (dst_address_a&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS) {
							slevel++;
							codegen.alloc_stack(slevel);
						}

						int src_address_b = _parse_assign_right_expression(codegen,on,slevel);
						if (src_address_b<0)
							return -1;




						codegen.opcodes.push_back(GDFunction::OPCODE_ASSIGN); // perform operator
						codegen.opcodes.push_back(dst_address_a); // argument 1
						codegen.opcodes.push_back(src_address_b); // argument 2 (unary only takes one parameter)
						return dst_address_a; //if anything, returns wathever was assigned or correct stack position

					}


				} break;
				case GDParser::OperatorNode::OP_EXTENDS: {

					ERR_FAIL_COND_V(on->arguments.size()!=2,false);


					int slevel = p_stack_level;

					int src_address_a = _parse_expression(codegen,on->arguments[0],slevel);
					if (src_address_a<0)
						return -1;

					if (src_address_a&GDFunction::ADDR_TYPE_STACK<<GDFunction::ADDR_BITS)
						slevel++; //uses stack for return, increase stack

					int src_address_b = _parse_expression(codegen,on->arguments[1],slevel);
					if (src_address_b<0)
						return -1;

					codegen.opcodes.push_back(GDFunction::OPCODE_EXTENDS_TEST); // perform operator
					codegen.opcodes.push_back(src_address_a); // argument 1
					codegen.opcodes.push_back(src_address_b); // argument 2 (unary only takes one parameter)

				} break;
				default: {


