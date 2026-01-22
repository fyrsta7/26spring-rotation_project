LogicalType RApiTypes::LogicalTypeFromRType(const RType &rtype, bool experimental) {
	switch (rtype.id()) {
	case RType::LOGICAL:
		return LogicalType::BOOLEAN;
	case RType::INTEGER:
		return LogicalType::INTEGER;
	case RType::NUMERIC:
		return LogicalType::DOUBLE;
	case RType::INTEGER64:
		return LogicalType::BIGINT;
	case RTypeId::FACTOR: {
		auto duckdb_levels = rtype.GetFactorLevels();
		return LogicalType::ENUM(duckdb_levels, rtype.GetFactorLevelsCount());
	}
	case RType::STRING:
		if (experimental) {
			return RStringsType::Get();
		} else {
			return LogicalType::VARCHAR;
		}
		break;
	case RType::TIMESTAMP:
		return LogicalType::TIMESTAMP;
	case RType::TIME_SECONDS:
	case RType::TIME_MINUTES:
	case RType::TIME_HOURS:
	case RType::TIME_DAYS:
	case RType::TIME_WEEKS:
		return LogicalType::TIME;
	case RType::TIME_SECONDS_INTEGER:
	case RType::TIME_MINUTES_INTEGER:
	case RType::TIME_HOURS_INTEGER:
	case RType::TIME_DAYS_INTEGER:
	case RType::TIME_WEEKS_INTEGER:
		return LogicalType::TIME;
	case RType::DATE:
		return LogicalType::DATE;
	case RType::DATE_INTEGER:
		return LogicalType::DATE;
	case RType::LIST_OF_NULLS:
	case RType::BLOB:
		return LogicalType::BLOB;
	case RTypeId::LIST:
		return LogicalType::LIST(RApiTypes::LogicalTypeFromRType(rtype.GetListChildType(), experimental));
	case RTypeId::STRUCT: {
		child_list_t<LogicalType> children;
		for (auto child : rtype.GetStructChildTypes()) {
			children.push_back(
			    std::make_pair(child.first, RApiTypes::LogicalTypeFromRType(child.second, experimental)));
		}
		if (children.size() == 0) {
			cpp11::stop("rapi_execute: Packed column must have at least one column");
		}
		return LogicalType::STRUCT(std::move(children));
	}

	default:
		cpp11::stop("rapi_execute: Can't convert R type to logical type");
	}
}
