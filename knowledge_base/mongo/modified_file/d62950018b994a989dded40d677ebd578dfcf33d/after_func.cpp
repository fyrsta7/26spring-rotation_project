 * for all of the code used other than as permitted herein. If you modify
 * file(s) with this exception, you may extend this exception to your
 * version of the file(s), but you are not obligated to do so. If you do not
 * wish to do so, delete this exception statement from your version. If you
 * delete this exception statement from all source files in the program,
 * then also delete it in the license file.
 */

#include "mongo/platform/basic.h"

#include "mongo/scripting/mozjs/db.h"

#include "mongo/db/namespace_string.h"
#include "mongo/db/operation_context.h"
#include "mongo/scripting/mozjs/idwrapper.h"
#include "mongo/scripting/mozjs/implscope.h"
#include "mongo/scripting/mozjs/objectwrapper.h"
#include "mongo/scripting/mozjs/valuereader.h"
#include "mongo/scripting/mozjs/valuewriter.h"
#include "mongo/s/d_state.h"

namespace mongo {
namespace mozjs {

const char* const DBInfo::className = "DB";

void DBInfo::getProperty(JSContext* cx,
                         JS::HandleObject obj,
                         JS::HandleId id,
                         JS::MutableHandleValue vp) {
    JS::RootedObject parent(cx);
    if (!JS_GetPrototype(cx, obj, &parent))
        uasserted(ErrorCodes::JSInterpreterFailure, "Couldn't get prototype");

    auto scope = getScope(cx);

    ObjectWrapper parentWrapper(cx, parent);

    // 2nd look into real values, may be cached collection object
    if (!vp.isUndefined()) {
        if (vp.isObject()) {
            ObjectWrapper o(cx, vp);

            if (o.hasField("_fullName")) {
                auto opContext = scope->getOpContext();

                // need to check every time that the collection did not get sharded
                if (opContext &&
                    haveLocalShardingInfo(opContext->getClient(), o.getString("_fullName")))
                    uasserted(ErrorCodes::BadValue, "can't use sharded collection from db.eval");
            }
        }

        return;
    } else if (parentWrapper.hasField(id)) {
        parentWrapper.getValue(id, vp);
        return;
    }

    std::string sname = IdWrapper(cx, id).toString();
    if (sname.length() == 0 || sname[0] == '_') {
        // if starts with '_' we dont return collection, one must use getCollection()
        return;
