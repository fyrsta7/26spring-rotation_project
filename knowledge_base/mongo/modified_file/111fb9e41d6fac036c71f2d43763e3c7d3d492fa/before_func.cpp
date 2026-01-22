#include "util/mongoutils/str.h"

namespace mongo {
    using namespace mongoutils;

    string Document::idName("_id");

    intrusive_ptr<Document> Document::createFromBsonObj(
        BSONObj *pBsonObj, const DependencyTracker *pDependencies) {
        intrusive_ptr<Document> pDocument(
            new Document(pBsonObj, pDependencies));
        return pDocument;
    }

    Document::Document(BSONObj *pBsonObj,
                       const DependencyTracker *pDependencies):
        vFieldName(),
        vpValue() {
