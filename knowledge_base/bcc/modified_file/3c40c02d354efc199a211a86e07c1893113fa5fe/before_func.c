
struct Key {
  unsigned char p[255];
};

struct Leaf {
  // Not really needed in this example
  unsigned char p[4];
};

BPF_HASH(cache, struct Key, struct Leaf, 128);

int dns_matching(struct __sk_buff *skb)
{
  u8 *cursor = 0;
  struct Key key = {};
  // Check of ethernet/IP frame.
  struct ethernet_t *ethernet = cursor_advance(cursor, sizeof(*ethernet));
  if(ethernet->type == ETH_P_IP) {

    // Check for UDP.
    struct ip_t *ip = cursor_advance(cursor, sizeof(*ip));
    u16 hlen_bytes = ip->hlen << 2;
    if(ip->nextp == IPPROTO_UDP) {

      // Check for Port 53, DNS packet.
      struct udp_t *udp = cursor_advance(cursor, sizeof(*udp));
      if(udp->dport == 53){

        // Our Cursor + the length of our udp packet - size of the udp header
        // - the two 16bit values for QTYPE and QCLASS.
        u8 *sentinel = cursor + udp->length - sizeof(*udp) - 4;

        struct dns_hdr_t *dns_hdr = cursor_advance(cursor, sizeof(*dns_hdr));

        // Do nothing if packet is not a request.
        if((dns_hdr->flags >>15) != 0) {
          // Exit if this packet is not a request.
          return -1;
        }

        u16 i = 0;
        struct dns_char_t *c;
        // This unroll worked not in latest BCC version.
        for(i = 0; i<255;i++){
          if (cursor == sentinel) goto end; c = cursor_advance(cursor, 1); key.p[i] = c->c;
        }
        end:
        {}

        struct Leaf * lookup_leaf = cache.lookup(&key);

