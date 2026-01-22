#include "bittorrent.h"
#include "misc.h"
#include "downloadThread.h"
#include "deleteThread.h"

#include <libtorrent/extensions/metadata_transfer.hpp>
#include <libtorrent/extensions/ut_pex.hpp>
#include <libtorrent/entry.hpp>
#include <libtorrent/bencode.hpp>
#include <libtorrent/identify_client.hpp>
#include <libtorrent/alert_types.hpp>
#include <libtorrent/torrent_info.hpp>
#include <boost/filesystem/exception.hpp>

#define ETAS_MAX_VALUES 3
#define ETA_REFRESH_INTERVAL 10000
#define MAX_TRACKER_ERRORS 2

// Main constructor
bittorrent::bittorrent() : timerScan(0), DHTEnabled(false), preAllocateAll(false), addInPause(false), maxConnecsPerTorrent(500), maxUploadsPerTorrent(4), max_ratio(-1) {
  // To avoid some exceptions
  fs::path::default_name_check(fs::no_check);
  // Creating bittorrent session
  s = new session(fingerprint("qB", VERSION_MAJOR, VERSION_MINOR, VERSION_BUGFIX, 0));
  // Set severity level of libtorrent session
  s->set_severity_level(alert::info);
