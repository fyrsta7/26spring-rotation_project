*    As a special exception, the copyright holders give permission to link the
*    code of portions of this program with the OpenSSL library under certain
*    conditions as described in each individual source file and distribute
*    linked combinations including the program with the OpenSSL library. You
*    must comply with the GNU Affero General Public License in all respects for
*    all of the code used other than as permitted herein. If you modify file(s)
*    with this exception, you may extend this exception to your version of the
*    file(s), but you are not obligated to do so. If you do not wish to do so,
*    delete this exception statement from your version. If you delete this
*    exception statement from all source files in the program, then also delete
*    it in the license file.
*/

#include "mongo/platform/basic.h"

#include "mongo/db/background.h"
#include "mongo/db/client.h"
#include "mongo/db/commands.h"
#include "mongo/db/index_builder.h"
#include "mongo/db/query/internal_plans.h"
#include "mongo/db/query/new_find.h"
#include "mongo/db/repl/oplog.h"
#include "mongo/db/operation_context_impl.h"

namespace mongo {

    Status cloneCollectionAsCapped( OperationContext* txn,
                                    Database* db,
                                    const string& shortFrom,
                                    const string& shortTo,
                                    double size,
                                    bool temp,
                                    bool logForReplication ) {

        string fromNs = db->name() + "." + shortFrom;
        string toNs = db->name() + "." + shortTo;

        Collection* fromCollection = db->getCollection( txn, fromNs );
        if ( !fromCollection )
            return Status( ErrorCodes::NamespaceNotFound,
                           str::stream() << "source collection " << fromNs <<  " does not exist" );

        if ( db->getCollection( txn, toNs ) )
            return Status( ErrorCodes::NamespaceExists, "to collection already exists" );

        // create new collection
        {
            Client::Context ctx(txn,  toNs );
            BSONObjBuilder spec;
            spec.appendBool( "capped", true );
            spec.append( "size", size );
            if ( temp )
                spec.appendBool( "temp", true );

            Status status = userCreateNS( txn, ctx.db(), toNs, spec.done(), logForReplication );
            if ( !status.isOK() )
                return status;
        }

        Collection* toCollection = db->getCollection( txn, toNs );
        invariant( toCollection ); // we created above

        // how much data to ignore because it won't fit anyway
        // datasize and extentSize can't be compared exactly, so add some padding to 'size'

        long long allocatedSpaceGuess =
            std::max( static_cast<long long>(size * 2),
                      static_cast<long long>(toCollection->getRecordStore()->storageSize(txn) * 2));

        long long excessSize = fromCollection->dataSize() - allocatedSpaceGuess;

        scoped_ptr<PlanExecutor> exec( InternalPlanner::collectionScan(txn,
                                                                       fromNs,
                                                                       fromCollection,
                                                                       InternalPlanner::FORWARD ) );


        while ( true ) {
