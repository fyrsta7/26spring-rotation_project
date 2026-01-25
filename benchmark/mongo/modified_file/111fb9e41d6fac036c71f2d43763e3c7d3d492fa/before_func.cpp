    Document::Document(BSONObj *pBsonObj,
                       const DependencyTracker *pDependencies):
        vFieldName(),
        vpValue() {
        BSONObjIterator bsonIterator(pBsonObj->begin());
        while(bsonIterator.more()) {
            BSONElement bsonElement(bsonIterator.next());
            string fieldName(bsonElement.fieldName());

            // LATER check pDependencies
            // LATER grovel through structures???
            intrusive_ptr<const Value> pValue(
                Value::createFromBsonElement(&bsonElement));

            vFieldName.push_back(fieldName);
            vpValue.push_back(pValue);
        }
    }