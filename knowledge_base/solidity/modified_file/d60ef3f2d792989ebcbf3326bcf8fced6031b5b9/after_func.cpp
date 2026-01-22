					m_context << eth::Instruction::DUP1;
					m_context << u256(type->getCalldataEncodedSize()) << eth::Instruction::ADD;
				}
			}
			else
			{
				solAssert(arrayType.location() == ReferenceType::Location::Memory, "");
				CompilerUtils(m_context).fetchFreeMemoryPointer();
				CompilerUtils(m_context).storeInMemoryDynamic(*type);
				CompilerUtils(m_context).storeFreeMemoryPointer();
			}
		}
		else
		{
			solAssert(!type->isDynamicallySized(), "Unknown dynamically sized type: " + type->toString());
			CompilerUtils(m_context).loadFromMemoryDynamic(*type, !_fromMemory, true);
		}
	}
	m_context << eth::Instruction::POP;
}

