    int         i_channel;                         /* current channel number */
    mtime_t     last_change;                             /* last change date */
} input_channel_t;

/*****************************************************************************
 * Local prototypes
 *****************************************************************************/
static int GetMacAddress   ( int i_fd, char *psz_mac );
#ifdef WIN32
static int GetAdapterInfo  ( int i_adapter, char *psz_string );
#endif

/*****************************************************************************
 * network_ChannelCreate: initialize global channel method data
 *****************************************************************************
 * Initialize channel input method global data. This function should be called
 * once before any input thread is created or any call to other
 * input_Channel*() function is attempted.
 *****************************************************************************/
int network_ChannelCreate( void )
{
#if !defined( SYS_LINUX ) && !defined( WIN32 )
    intf_ErrMsg( "channel warning: VLAN-based channels are not supported"
                 " under this architecture" );
#endif

    /* Allocate structure */
    p_main->p_channel = malloc( sizeof( input_channel_t ) );
    if( p_main->p_channel == NULL )
    {
        intf_ErrMsg( "network error: could not create channel bank" );
        return( -1 );
    }

    /* Initialize structure */
    p_main->p_channel->i_channel   = 0;
    p_main->p_channel->last_change = 0;

    intf_WarnMsg( 2, "network: channels initialized" );
    return( 0 );
}

/*****************************************************************************
 * network_ChannelJoin: join a channel
 *****************************************************************************
 * This function will try to join a channel. If the relevant interface is
 * already on the good channel, nothing will be done. Else, and if possible
 * (if the interface is not locked), the channel server will be contacted
 * and a change will be requested. The function will block until the change
 * is effective. Note that once a channel is no more used, its interface
 * should be unlocked using input_ChannelLeave().
 * Non 0 will be returned in case of error.
 *****************************************************************************/
int network_ChannelJoin( int i_channel )
{
#define VLCS_VERSION 13
#define MESSAGE_LENGTH 256

    struct module_s *   p_network;
    char *              psz_network = NULL;
    network_socket_t    socket_desc;
    char psz_mess[ MESSAGE_LENGTH ];
    char psz_mac[ 40 ];
    int i_fd, i_port;
    char *psz_vlcs;
    struct timeval delay;
    fd_set fds;

    if( !config_GetIntVariable( "network-channel" ) )
    {
        intf_ErrMsg( "network: channels disabled, to enable them, use the"
                     "--channels option" );
        return -1;
    }

    /* If last change is too recent, wait a while */
    if( mdate() - p_main->p_channel->last_change < INPUT_CHANNEL_CHANGE_DELAY )
    {
        intf_WarnMsg( 2, "network: waiting before changing channel" );
        /* XXX Isn't this completely brain-damaged ??? -- Sam */
        /* Yes it is. I don't think this is still justified with the new
         * vlanserver --Meuuh */
        mwait( p_main->p_channel->last_change + INPUT_CHANNEL_CHANGE_DELAY );
    }

    if( config_GetIntVariable( "ipv4" ) )
    {
        psz_network = "ipv4";
    }
    if( config_GetIntVariable( "ipv6" ) )
    {
        psz_network = "ipv6";
    }

    /* Getting information about the channel server */
    if( !(psz_vlcs = config_GetPszVariable( "channel-server" )) )
    {
        intf_ErrMsg( "network: configuration variable channel_server empty" );
        return -1;
    }

    i_port = config_GetIntVariable( "channel-port" );

    intf_WarnMsg( 5, "channel: connecting to %s:%d",
                     psz_vlcs, i_port );

    /* Prepare the network_socket_t structure */
    socket_desc.i_type = NETWORK_UDP;
    socket_desc.psz_bind_addr = "";
    socket_desc.i_bind_port = 4321;
    socket_desc.psz_server_addr = psz_vlcs;
    socket_desc.i_server_port = i_port;

    /* Find an appropriate network module */
    p_network = module_Need( MODULE_CAPABILITY_NETWORK, psz_network,
                             &socket_desc );
    if( p_network == NULL )
    {
        return( -1 );
    }
    module_Unneed( p_network );

    free( psz_vlcs ); /* Do we really need this ? -- Meuuh */
    i_fd = socket_desc.i_handle;

    /* Look for the interface MAC address */
    if( GetMacAddress( i_fd, psz_mac ) )
    {
        intf_ErrMsg( "network error: failed getting MAC address" );
        close( i_fd );
        return -1;
    }

    intf_WarnMsg( 6, "network: MAC address is %s", psz_mac );

    /* Build the message */
    sprintf( psz_mess, "%d %u %lu %s \n", i_channel, VLCS_VERSION,
                       (unsigned long)(mdate() / (u64)1000000),
                       psz_mac );

    /* Send the message */
    send( i_fd, psz_mess, MESSAGE_LENGTH, 0 );

    intf_WarnMsg( 2, "network: attempting to join channel %d", i_channel );

    /* We have changed channels ! (or at least, we tried) */
    p_main->p_channel->last_change = mdate();
    p_main->p_channel->i_channel = i_channel;

    /* Wait 5 sec for an answer from the server */
    delay.tv_sec = 5;
    delay.tv_usec = 0;
    FD_ZERO( &fds );
    FD_SET( i_fd, &fds );

    switch( select( i_fd + 1, &fds, NULL, NULL, &delay ) )
    {
        case 0:
