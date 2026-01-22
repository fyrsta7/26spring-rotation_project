#include <vlc_vout_window.h>                /* VOUT_ events */

#define VLC_REFERENCE_SCALE_FACTOR 96.

using  namespace vlc::playlist;

// #define DEBUG_INTF

/* Callback prototypes */
static int PopupMenuCB( vlc_object_t *p_this, const char *psz_variable,
                        vlc_value_t old_val, vlc_value_t new_val, void *param );
static int IntfShowCB( vlc_object_t *p_this, const char *psz_variable,
                       vlc_value_t old_val, vlc_value_t new_val, void *param );
static int IntfBossCB( vlc_object_t *p_this, const char *psz_variable,
                       vlc_value_t old_val, vlc_value_t new_val, void *param );
static int IntfRaiseMainCB( vlc_object_t *p_this, const char *psz_variable,
                           vlc_value_t old_val, vlc_value_t new_val,
                           void *param );

const QEvent::Type MainInterface::ToolbarsNeedRebuild =
        (QEvent::Type)QEvent::registerEventType();

MainInterface::MainInterface(intf_thread_t *_p_intf , QWidget* parent, Qt::WindowFlags flags)
    : QVLCMW( _p_intf, parent, flags )
{
    /* Variables initialisation */
    lastWinScreen        = NULL;
    sysTray              = NULL;
    cryptedLabel         = NULL;

    b_hideAfterCreation  = false; // --qt-start-minimized
    playlistVisible      = false;
    playlistWidthFactor  = 4.0;
    b_interfaceFullScreen= false;
    i_kc_offset          = false;

    /**
     *  Configuration and settings
     *  Pre-building of interface
     **/

    /* Are we in the enhanced always-video mode or not ? */
    b_minimalView = var_InheritBool( p_intf, "qt-minimal-view" );

    /* Do we want anoying popups or not */
    i_notificationSetting = var_InheritInteger( p_intf, "qt-notification" );

    /* */
    m_intfUserScaleFactor = var_InheritFloat(p_intf, "qt-interface-scale");
    if (m_intfUserScaleFactor == -1)
        m_intfUserScaleFactor = getSettings()->value( "MainWindow/interface-scale", 1.0).toFloat();
    winId(); //force window creation
    QWindow* window = windowHandle();
    if (window)
        connect(window, &QWindow::screenChanged, this, &MainInterface::updateIntfScaleFactor);
    updateIntfScaleFactor();

    /* Get the available interfaces */
    m_extraInterfaces = new VLCVarChoiceModel(p_intf, "intf-add", this);

    /* Set the other interface settings */
    settings = getSettings();

    /* playlist settings */
    b_playlistDocked = getSettings()->value( "MainWindow/pl-dock-status", true ).toBool();
    playlistVisible  = getSettings()->value( "MainWindow/playlist-visible", false ).toBool();
    playlistWidthFactor = getSettings()->value( "MainWindow/playlist-width-factor", 4.0 ).toDouble();

    m_showRemainingTime = getSettings()->value( "MainWindow/ShowRemainingTime", false ).toBool();

    /* Should the UI stays on top of other windows */
    b_interfaceOnTop = var_InheritBool( p_intf, "video-on-top" );

    QString platformName = QGuiApplication::platformName();

#ifdef QT5_HAS_WAYLAND
    b_hasWayland = platformName.startsWith(QLatin1String("wayland"), Qt::CaseInsensitive);
#endif

    /**************************
     *  UI and Widgets design
     **************************/

    /* Main settings */
    setFocusPolicy( Qt::StrongFocus );
    setAcceptDrops( true );

    /*********************************
     * Create the Systray Management *
     *********************************/
    //postpone systray initialisation to speedup starting time
    QMetaObject::invokeMethod(this, &MainInterface::initSystray, Qt::QueuedConnection);

    /*************************************************************
     * Connect the input manager to the GUI elements it manages  *
     * Beware initSystray did some connects on input manager too *
     *************************************************************/
    /**
     * Connects on nameChanged()
     * Those connects are different because options can impeach them to trigger.
     **/
