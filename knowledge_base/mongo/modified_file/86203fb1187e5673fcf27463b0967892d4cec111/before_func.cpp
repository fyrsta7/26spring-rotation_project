        if (proto->IsRegExp())
            v8ToMongoRegex(b, elementName, obj);
        else if (proto->IsObject() &&
                 proto->ToObject()->HasRealNamedProperty(v8::String::New("isObjectId")))
            v8ToMongoObjectID(b, elementName, obj);
        else if (!obj->GetHiddenValue(v8::String::New("__NumberLong")).IsEmpty())
            v8ToMongoNumberLong(b, elementName, obj);
        else if (!obj->GetHiddenValue(v8::String::New("__NumberInt")).IsEmpty())
            b.append(elementName,
                     obj->GetHiddenValue(v8::String::New("__NumberInt"))->Int32Value());
        else if (!value->ToObject()->GetHiddenValue(v8::String::New("__DBPointer")).IsEmpty())
            v8ToMongoDBRef(b, elementName, obj);
        else if (!value->ToObject()->GetHiddenValue(v8::String::New("__BinData")).IsEmpty())
            v8ToMongoBinData(b, elementName, obj);
        else {
            // nested object or array
            BSONObj sub = v8ToMongo(obj, depth);
            b.append(elementName, sub);
        }
    }

    void V8Scope::v8ToMongoElement(BSONObjBuilder & b, const string& sname,
                                   v8::Handle<v8::Value> value, int depth,
                                   BSONObj* originalParent) {
        if (value->IsString()) {
            b.append(sname, toSTLString(value));
            return;
        }
        if (value->IsFunction()) {
            uassert(16716, "cannot convert native function to BSON",
                    !value->ToObject()->Has(v8StringData("_v8_function")));
            b.appendCode(sname, toSTLString(value));
            return;
        }
        if (value->IsNumber()) {
            v8ToMongoNumber(b, sname, value, originalParent);
            return;
        }
        if (value->IsArray()) {
            BSONObj sub = v8ToMongo(value->ToObject(), depth);
            b.appendArray(sname, sub);
            return;
        }
        if (value->IsDate()) {
            long long dateval = (long long)(v8::Date::Cast(*value)->NumberValue());
            b.appendDate(sname, Date_t((unsigned long long) dateval));
            return;
        }
        if (value->IsExternal())
