 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * REVISIONS:
 *     09-Sep-2003 vizzini - Created
 *     10-Oct-2004 navaraf - Fix receive to work on VMware adapters (
 *                           need to set busmaster bit on PCI).
 *                         - Indicate receive completition.
 *                         - Implement packet transmitting.
 *                         - Don't read slot number from registry and
 *                           report itself as NDIS 5.0 miniport.
 *     11-Oct-2004 navaraf - Fix nasty bugs in halt code path.
 *     17-Oct-2004 navaraf - Add multicast support.
 *                         - Add media state detection support.
 *                         - Protect the adapter context with spinlock
 *                           and move code talking to card to inside
 *                           NdisMSynchronizeWithInterrupt calls where
 *                           necessary.
 *
 * NOTES:
 *     - this assumes a 32-bit machine
 */

#include <ndis.h>
#include "pci.h"
#include "pcnethw.h"
#include "pcnet.h"

#define NDEBUG
#include <debug.h>

NTSTATUS
NTAPI
DriverEntry(
    IN PDRIVER_OBJECT DriverObject,
    IN PUNICODE_STRING RegistryPath);

static VOID
NTAPI
MiniportHandleInterrupt(
    IN NDIS_HANDLE MiniportAdapterContext)
/*
 * FUNCTION: Handle an interrupt if told to by MiniportISR
 * ARGUMENTS:
 *     MiniportAdapterContext: context specified to NdisMSetAttributes
 * NOTES:
 *     - Called by NDIS at DISPATCH_LEVEL
 */
{
  PADAPTER Adapter = (PADAPTER)MiniportAdapterContext;
  USHORT Data;

  DPRINT("Called\n");

  ASSERT_IRQL_EQUAL(DISPATCH_LEVEL);

  NdisDprAcquireSpinLock(&Adapter->Lock);

  NdisRawWritePortUshort(Adapter->PortOffset + RAP, CSR0);
  NdisRawReadPortUshort(Adapter->PortOffset + RDP, &Data);

  DPRINT("CSR0 is 0x%x\n", Data);

  while(Data & CSR0_INTR)
    {
      /* Clear interrupt flags early to avoid race conditions. */
      NdisRawWritePortUshort(Adapter->PortOffset + RDP, Data);

      if(Data & CSR0_ERR)
        {
          DPRINT("error: %x\n", Data & (CSR0_MERR|CSR0_BABL|CSR0_CERR|CSR0_MISS));
          if (Data & CSR0_CERR)
            Adapter->Statistics.XmtCollisions++;
        }
      if(Data & CSR0_IDON)
        {
          DPRINT("IDON\n");
        }
      if(Data & CSR0_RINT)
        {
          DPRINT("receive interrupt\n");

          while(1)
            {
              PRECEIVE_DESCRIPTOR Descriptor = Adapter->ReceiveDescriptorRingVirt + Adapter->CurrentReceiveDescriptorIndex;
              PCHAR Buffer;
              ULONG ByteCount;

              if(Descriptor->FLAGS & RD_OWN)
                {
                  DPRINT("no more receive descriptors to process\n");
                  break;
                }

              if(Descriptor->FLAGS & RD_ERR)
                {
                  DPRINT("receive descriptor error: 0x%x\n", Descriptor->FLAGS);
                  if (Descriptor->FLAGS & RD_BUFF)
                    Adapter->Statistics.RcvBufferErrors++;
                  if (Descriptor->FLAGS & RD_CRC)
                    Adapter->Statistics.RcvCrcErrors++;
                  if (Descriptor->FLAGS & RD_OFLO)
                    Adapter->Statistics.RcvOverflowErrors++;
                  if (Descriptor->FLAGS & RD_FRAM)
                    Adapter->Statistics.RcvFramingErrors++;
                  break;
                }

              if(!((Descriptor->FLAGS & RD_STP) && (Descriptor->FLAGS & RD_ENP)))
                {
                  DPRINT("receive descriptor not start&end: 0x%x\n", Descriptor->FLAGS);
                  break;
                }

              Buffer = Adapter->ReceiveBufferPtrVirt + Adapter->CurrentReceiveDescriptorIndex * BUFFER_SIZE;
              ByteCount = Descriptor->MCNT & 0xfff;

              DPRINT("Indicating a %d-byte packet (index %d)\n", ByteCount, Adapter->CurrentReceiveDescriptorIndex);

              NdisMEthIndicateReceive(Adapter->MiniportAdapterHandle, 0, Buffer, 14, Buffer+14, ByteCount-14, ByteCount-14);
              NdisMEthIndicateReceiveComplete(Adapter->MiniportAdapterHandle);

              RtlZeroMemory(Descriptor, sizeof(RECEIVE_DESCRIPTOR));
              Descriptor->RBADR =
                  (ULONG)(Adapter->ReceiveBufferPtrPhys + Adapter->CurrentReceiveDescriptorIndex * BUFFER_SIZE);
              Descriptor->BCNT = (-BUFFER_SIZE) | 0xf000;
              Descriptor->FLAGS |= RD_OWN;

              Adapter->CurrentReceiveDescriptorIndex++;
              Adapter->CurrentReceiveDescriptorIndex %= NUMBER_OF_BUFFERS;

              Adapter->Statistics.RcvGoodFrames++;
            }
        }
      if(Data & CSR0_TINT)
        {
          PTRANSMIT_DESCRIPTOR Descriptor;

          DPRINT("transmit interrupt\n");

          while (Adapter->CurrentTransmitStartIndex !=
                 Adapter->CurrentTransmitEndIndex)
            {
              Descriptor = Adapter->TransmitDescriptorRingVirt + Adapter->CurrentTransmitStartIndex;

              DPRINT("buffer %d flags %x flags2 %x\n",
                     Adapter->CurrentTransmitStartIndex,
                     Descriptor->FLAGS, Descriptor->FLAGS2);

              if (Descriptor->FLAGS & TD1_OWN)
                {
                  DPRINT("non-TXed buffer\n");
                  break;
                }

              if (Descriptor->FLAGS & TD1_STP)
                {
                  if (Descriptor->FLAGS & TD1_ONE)
                    Adapter->Statistics.XmtOneRetry++;
                  else if (Descriptor->FLAGS & TD1_MORE)
