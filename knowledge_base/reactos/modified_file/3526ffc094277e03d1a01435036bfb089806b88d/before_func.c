                cursory++;
                /* Fall back */
            case '\r':
                cursorx = 0;
                break;

            case '\t':
            {
                offset = TAB_WIDTH - (cursorx % TAB_WIDTH);
                while (offset--)
                {
                    vidmem[(cursorx + cursory * columns) * 2] = ' ';
                    cursorx++;
                    if (cursorx >= columns)
                    {
                        cursorx = 0;
                        cursory++;
                        /* We jumped to the next line, stop there */
                        break;
                    }
                }
                break;
            }

            default:
            {
                offset = cursorx + cursory * columns;
                vidmem[offset * 2] = *pch;
                vidmem[offset * 2 + 1] = (char)DeviceExtension->CharAttribute;
                cursorx++;
                if (cursorx >= columns)
                {
                    cursorx = 0;
                    cursory++;
                }
                break;
            }
            }

            /* Scroll up the contents of the screen if we are at the end */
            if (cursory >= rows)
            {
                PUSHORT LinePtr;

                RtlCopyMemory(vidmem,
                              &vidmem[columns * 2],
                              columns * (rows - 1) * 2);

                LinePtr = (PUSHORT)&vidmem[columns * (rows - 1) * 2];

                for (j = 0; j < columns; j++)
                {
                    LinePtr[j] = DeviceExtension->CharAttribute << 8;
                }
                cursory = rows - 1;
                for (j = 0; j < columns; j++)
                {
                    offset = j + cursory * columns;
                    vidmem[offset * 2] = ' ';
                    vidmem[offset * 2 + 1] = (char)DeviceExtension->CharAttribute;
                }
            }
        }
    }

    /* Set the cursor position */
    ASSERT((0 <= cursorx) && (cursorx < DeviceExtension->Columns));
    ASSERT((0 <= cursory) && (cursory < DeviceExtension->Rows));
    DeviceExtension->CursorX = cursorx;
    DeviceExtension->CursorY = cursory;
    ScrSetCursor(DeviceExtension);

    Status = STATUS_SUCCESS;

    Irp->IoStatus.Status = Status;
    IoCompleteRequest(Irp, IO_VIDEO_INCREMENT);

    return Status;
}

static DRIVER_DISPATCH ScrIoControl;
static NTSTATUS
NTAPI
ScrIoControl(
    _In_ PDEVICE_OBJECT DeviceObject,
    _In_ PIRP Irp)
{
    NTSTATUS Status;
    PIO_STACK_LOCATION stk = IoGetCurrentIrpStackLocation(Irp);
    PDEVICE_EXTENSION DeviceExtension = DeviceObject->DeviceExtension;

    switch (stk->Parameters.DeviceIoControl.IoControlCode)
    {
        case IOCTL_CONSOLE_RESET_SCREEN:
        {
            BOOLEAN Enable;

            /* Validate input buffer */
            if (stk->Parameters.DeviceIoControl.InputBufferLength < sizeof(ULONG))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            Enable = !!*(PULONG)Irp->AssociatedIrp.SystemBuffer;

            /* Fully enable or disable the screen */
            Status = (ScrResetScreen(DeviceExtension, TRUE, Enable)
                        ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL);
            Irp->IoStatus.Information = 0;
            break;
        }

        case IOCTL_CONSOLE_GET_SCREEN_BUFFER_INFO:
        {
            PCONSOLE_SCREEN_BUFFER_INFO pcsbi;
            USHORT rows = DeviceExtension->Rows;
            USHORT columns = DeviceExtension->Columns;

            /* Validate output buffer */
            if (stk->Parameters.DeviceIoControl.OutputBufferLength < sizeof(CONSOLE_SCREEN_BUFFER_INFO))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            pcsbi = (PCONSOLE_SCREEN_BUFFER_INFO)Irp->AssociatedIrp.SystemBuffer;
            RtlZeroMemory(pcsbi, sizeof(CONSOLE_SCREEN_BUFFER_INFO));

            pcsbi->dwSize.X = columns;
            pcsbi->dwSize.Y = rows;

            pcsbi->dwCursorPosition.X = DeviceExtension->CursorX;
            pcsbi->dwCursorPosition.Y = DeviceExtension->CursorY;

            pcsbi->wAttributes = DeviceExtension->CharAttribute;

            pcsbi->srWindow.Left   = 0;
            pcsbi->srWindow.Right  = columns - 1;
            pcsbi->srWindow.Top    = 0;
            pcsbi->srWindow.Bottom = rows - 1;

            pcsbi->dwMaximumWindowSize.X = columns;
            pcsbi->dwMaximumWindowSize.Y = rows;

            Irp->IoStatus.Information = sizeof(CONSOLE_SCREEN_BUFFER_INFO);
            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_SET_SCREEN_BUFFER_INFO:
        {
            PCONSOLE_SCREEN_BUFFER_INFO pcsbi;

            /* Validate input buffer */
            if (stk->Parameters.DeviceIoControl.InputBufferLength < sizeof(CONSOLE_SCREEN_BUFFER_INFO))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            pcsbi = (PCONSOLE_SCREEN_BUFFER_INFO)Irp->AssociatedIrp.SystemBuffer;

            if ( pcsbi->dwCursorPosition.X < 0 || pcsbi->dwCursorPosition.X >= DeviceExtension->Columns ||
                 pcsbi->dwCursorPosition.Y < 0 || pcsbi->dwCursorPosition.Y >= DeviceExtension->Rows )
            {
                Irp->IoStatus.Information = 0;
                Status = STATUS_INVALID_PARAMETER;
                break;
            }

            DeviceExtension->CharAttribute = pcsbi->wAttributes;

            /* Set the cursor position */
            ASSERT((0 <= pcsbi->dwCursorPosition.X) && (pcsbi->dwCursorPosition.X < DeviceExtension->Columns));
            ASSERT((0 <= pcsbi->dwCursorPosition.Y) && (pcsbi->dwCursorPosition.Y < DeviceExtension->Rows));
            DeviceExtension->CursorX = pcsbi->dwCursorPosition.X;
            DeviceExtension->CursorY = pcsbi->dwCursorPosition.Y;
            if (DeviceExtension->Enabled)
                ScrSetCursor(DeviceExtension);

            Irp->IoStatus.Information = 0;
            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_GET_CURSOR_INFO:
        {
            PCONSOLE_CURSOR_INFO pcci;

            /* Validate output buffer */
            if (stk->Parameters.DeviceIoControl.OutputBufferLength < sizeof(CONSOLE_CURSOR_INFO))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            pcci = (PCONSOLE_CURSOR_INFO)Irp->AssociatedIrp.SystemBuffer;
            RtlZeroMemory(pcci, sizeof(CONSOLE_CURSOR_INFO));

            pcci->dwSize = DeviceExtension->CursorSize;
            pcci->bVisible = DeviceExtension->CursorVisible;

            Irp->IoStatus.Information = sizeof(CONSOLE_CURSOR_INFO);
            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_SET_CURSOR_INFO:
        {
            PCONSOLE_CURSOR_INFO pcci;

            /* Validate input buffer */
            if (stk->Parameters.DeviceIoControl.InputBufferLength < sizeof(CONSOLE_CURSOR_INFO))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            pcci = (PCONSOLE_CURSOR_INFO)Irp->AssociatedIrp.SystemBuffer;

            DeviceExtension->CursorSize = pcci->dwSize;
            DeviceExtension->CursorVisible = pcci->bVisible;
            if (DeviceExtension->Enabled)
                ScrSetCursorShape(DeviceExtension);

            Irp->IoStatus.Information = 0;
            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_GET_MODE:
        {
            PCONSOLE_MODE pcm;

            /* Validate output buffer */
            if (stk->Parameters.DeviceIoControl.OutputBufferLength < sizeof(CONSOLE_MODE))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            pcm = (PCONSOLE_MODE)Irp->AssociatedIrp.SystemBuffer;
            RtlZeroMemory(pcm, sizeof(CONSOLE_MODE));

            pcm->dwMode = DeviceExtension->Mode;

            Irp->IoStatus.Information = sizeof(CONSOLE_MODE);
            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_SET_MODE:
        {
            PCONSOLE_MODE pcm;

            /* Validate input buffer */
            if (stk->Parameters.DeviceIoControl.InputBufferLength < sizeof(CONSOLE_MODE))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            pcm = (PCONSOLE_MODE)Irp->AssociatedIrp.SystemBuffer;
            DeviceExtension->Mode = pcm->dwMode;

            Irp->IoStatus.Information = 0;
            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_FILL_OUTPUT_ATTRIBUTE:
        {
            POUTPUT_ATTRIBUTE Buf;
            PUCHAR vidmem;
            ULONG offset;
            ULONG dwCount;
            ULONG nMaxLength;

            /* Validate input and output buffers */
            if (stk->Parameters.DeviceIoControl.InputBufferLength  < sizeof(OUTPUT_ATTRIBUTE) ||
                stk->Parameters.DeviceIoControl.OutputBufferLength < sizeof(OUTPUT_ATTRIBUTE))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            Buf = (POUTPUT_ATTRIBUTE)Irp->AssociatedIrp.SystemBuffer;
            nMaxLength = Buf->nLength;

            Buf->dwTransfered = 0;
            Irp->IoStatus.Information = sizeof(OUTPUT_ATTRIBUTE);

            if ( Buf->dwCoord.X < 0 || Buf->dwCoord.X >= DeviceExtension->Columns ||
                 Buf->dwCoord.Y < 0 || Buf->dwCoord.Y >= DeviceExtension->Rows    ||
                 nMaxLength == 0 )
            {
                Status = STATUS_SUCCESS;
                break;
            }

            if (DeviceExtension->Enabled && DeviceExtension->VideoMemory)
            {
                vidmem = DeviceExtension->VideoMemory;
                offset = (Buf->dwCoord.X + Buf->dwCoord.Y * DeviceExtension->Columns) * 2 + 1;

                nMaxLength = min(nMaxLength,
                                 (DeviceExtension->Rows - Buf->dwCoord.Y)
                                    * DeviceExtension->Columns - Buf->dwCoord.X);

                for (dwCount = 0; dwCount < nMaxLength; dwCount++)
                {
                    vidmem[offset + (dwCount * 2)] = (char)Buf->wAttribute;
                }
                Buf->dwTransfered = dwCount;
            }

            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_READ_OUTPUT_ATTRIBUTE:
        {
            POUTPUT_ATTRIBUTE Buf;
            PUSHORT pAttr;
            PUCHAR vidmem;
            ULONG offset;
            ULONG dwCount;
            ULONG nMaxLength;

            /* Validate input buffer */
            if (stk->Parameters.DeviceIoControl.InputBufferLength < sizeof(OUTPUT_ATTRIBUTE))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            Buf = (POUTPUT_ATTRIBUTE)Irp->AssociatedIrp.SystemBuffer;
            Irp->IoStatus.Information = 0;

            /* Validate output buffer */
            if (stk->Parameters.DeviceIoControl.OutputBufferLength == 0)
            {
                Status = STATUS_SUCCESS;
                break;
            }
            ASSERT(Irp->MdlAddress);
            pAttr = MmGetSystemAddressForMdlSafe(Irp->MdlAddress, NormalPagePriority);
            if (pAttr == NULL)
            {
                Status = STATUS_INSUFFICIENT_RESOURCES;
                break;
            }

            if ( Buf->dwCoord.X < 0 || Buf->dwCoord.X >= DeviceExtension->Columns ||
                 Buf->dwCoord.Y < 0 || Buf->dwCoord.Y >= DeviceExtension->Rows )
            {
                Status = STATUS_SUCCESS;
                break;
            }

            nMaxLength = stk->Parameters.DeviceIoControl.OutputBufferLength;
            nMaxLength /= sizeof(USHORT);

            if (DeviceExtension->Enabled && DeviceExtension->VideoMemory)
            {
                vidmem = DeviceExtension->VideoMemory;
                offset = (Buf->dwCoord.X + Buf->dwCoord.Y * DeviceExtension->Columns) * 2 + 1;

                nMaxLength = min(nMaxLength,
                                 (DeviceExtension->Rows - Buf->dwCoord.Y)
                                    * DeviceExtension->Columns - Buf->dwCoord.X);

                for (dwCount = 0; dwCount < nMaxLength; dwCount++, pAttr++)
                {
                    *((PCHAR)pAttr) = vidmem[offset + (dwCount * 2)];
                }
                Irp->IoStatus.Information = dwCount * sizeof(USHORT);
            }

            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_WRITE_OUTPUT_ATTRIBUTE:
        {
            COORD dwCoord;
            PCOORD pCoord;
            PUSHORT pAttr;
            PUCHAR vidmem;
            ULONG offset;
            ULONG dwCount;
            ULONG nMaxLength;

            //
            // NOTE: For whatever reason no OUTPUT_ATTRIBUTE structure
            // is used for this IOCTL.
            //

            /* Validate output buffer */
            if (stk->Parameters.DeviceIoControl.OutputBufferLength < sizeof(COORD))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->MdlAddress);
            pCoord = MmGetSystemAddressForMdlSafe(Irp->MdlAddress, NormalPagePriority);
            if (pCoord == NULL)
            {
                Status = STATUS_INSUFFICIENT_RESOURCES;
                break;
            }
            /* Capture the input info data */
            dwCoord = *pCoord;

            nMaxLength = stk->Parameters.DeviceIoControl.OutputBufferLength - sizeof(COORD);
            nMaxLength /= sizeof(USHORT);

            Irp->IoStatus.Information = 0;

            if ( dwCoord.X < 0 || dwCoord.X >= DeviceExtension->Columns ||
                 dwCoord.Y < 0 || dwCoord.Y >= DeviceExtension->Rows    ||
                 nMaxLength == 0 )
            {
                Status = STATUS_SUCCESS;
                break;
            }

            pAttr = (PUSHORT)(pCoord + 1);

            if (DeviceExtension->Enabled && DeviceExtension->VideoMemory)
            {
                vidmem = DeviceExtension->VideoMemory;
                offset = (dwCoord.X + dwCoord.Y * DeviceExtension->Columns) * 2 + 1;

                nMaxLength = min(nMaxLength,
                                 (DeviceExtension->Rows - dwCoord.Y)
                                    * DeviceExtension->Columns - dwCoord.X);

                for (dwCount = 0; dwCount < nMaxLength; dwCount++, pAttr++)
                {
                    vidmem[offset + (dwCount * 2)] = *((PCHAR)pAttr);
                }
                Irp->IoStatus.Information = dwCount * sizeof(USHORT);
            }

            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_SET_TEXT_ATTRIBUTE:
        {
            /* Validate input buffer */
            if (stk->Parameters.DeviceIoControl.InputBufferLength < sizeof(USHORT))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            DeviceExtension->CharAttribute = *(PUSHORT)Irp->AssociatedIrp.SystemBuffer;

            Irp->IoStatus.Information = 0;
            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_FILL_OUTPUT_CHARACTER:
        {
            POUTPUT_CHARACTER Buf;
            PUCHAR vidmem;
            ULONG offset;
            ULONG dwCount;
            ULONG nMaxLength;

            /* Validate input and output buffers */
            if (stk->Parameters.DeviceIoControl.InputBufferLength  < sizeof(OUTPUT_CHARACTER) ||
                stk->Parameters.DeviceIoControl.OutputBufferLength < sizeof(OUTPUT_CHARACTER))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            Buf = (POUTPUT_CHARACTER)Irp->AssociatedIrp.SystemBuffer;
            nMaxLength = Buf->nLength;

            Buf->dwTransfered = 0;
            Irp->IoStatus.Information = sizeof(OUTPUT_CHARACTER);

            if ( Buf->dwCoord.X < 0 || Buf->dwCoord.X >= DeviceExtension->Columns ||
                 Buf->dwCoord.Y < 0 || Buf->dwCoord.Y >= DeviceExtension->Rows    ||
                 nMaxLength == 0 )
            {
                Status = STATUS_SUCCESS;
                break;
            }

            if (DeviceExtension->Enabled && DeviceExtension->VideoMemory)
            {
                vidmem = DeviceExtension->VideoMemory;
                offset = (Buf->dwCoord.X + Buf->dwCoord.Y * DeviceExtension->Columns) * 2;

                nMaxLength = min(nMaxLength,
                                 (DeviceExtension->Rows - Buf->dwCoord.Y)
                                    * DeviceExtension->Columns - Buf->dwCoord.X);

                for (dwCount = 0; dwCount < nMaxLength; dwCount++)
                {
                    vidmem[offset + (dwCount * 2)] = (char)Buf->cCharacter;
                }
                Buf->dwTransfered = dwCount;
            }

            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_READ_OUTPUT_CHARACTER:
        {
            POUTPUT_CHARACTER Buf;
            PCHAR pChar;
            PUCHAR vidmem;
            ULONG offset;
            ULONG dwCount;
            ULONG nMaxLength;

            /* Validate input buffer */
            if (stk->Parameters.DeviceIoControl.InputBufferLength < sizeof(OUTPUT_CHARACTER))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->AssociatedIrp.SystemBuffer);

            Buf = (POUTPUT_CHARACTER)Irp->AssociatedIrp.SystemBuffer;
            Irp->IoStatus.Information = 0;

            /* Validate output buffer */
            if (stk->Parameters.DeviceIoControl.OutputBufferLength == 0)
            {
                Status = STATUS_SUCCESS;
                break;
            }
            ASSERT(Irp->MdlAddress);
            pChar = MmGetSystemAddressForMdlSafe(Irp->MdlAddress, NormalPagePriority);
            if (pChar == NULL)
            {
                Status = STATUS_INSUFFICIENT_RESOURCES;
                break;
            }

            if ( Buf->dwCoord.X < 0 || Buf->dwCoord.X >= DeviceExtension->Columns ||
                 Buf->dwCoord.Y < 0 || Buf->dwCoord.Y >= DeviceExtension->Rows )
            {
                Status = STATUS_SUCCESS;
                break;
            }

            nMaxLength = stk->Parameters.DeviceIoControl.OutputBufferLength;

            if (DeviceExtension->Enabled && DeviceExtension->VideoMemory)
            {
                vidmem = DeviceExtension->VideoMemory;
                offset = (Buf->dwCoord.X + Buf->dwCoord.Y * DeviceExtension->Columns) * 2;

                nMaxLength = min(nMaxLength,
                                 (DeviceExtension->Rows - Buf->dwCoord.Y)
                                    * DeviceExtension->Columns - Buf->dwCoord.X);

                for (dwCount = 0; dwCount < nMaxLength; dwCount++, pChar++)
                {
                    *pChar = vidmem[offset + (dwCount * 2)];
                }
                Irp->IoStatus.Information = dwCount * sizeof(CHAR);
            }

            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_WRITE_OUTPUT_CHARACTER:
        {
            COORD dwCoord;
            PCOORD pCoord;
            PCHAR pChar;
            PUCHAR vidmem;
            ULONG offset;
            ULONG dwCount;
            ULONG nMaxLength;

            //
            // NOTE: For whatever reason no OUTPUT_CHARACTER structure
            // is used for this IOCTL.
            //

            /* Validate output buffer */
            if (stk->Parameters.DeviceIoControl.OutputBufferLength < sizeof(COORD))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->MdlAddress);
            pCoord = MmGetSystemAddressForMdlSafe(Irp->MdlAddress, NormalPagePriority);
            if (pCoord == NULL)
            {
                Status = STATUS_INSUFFICIENT_RESOURCES;
                break;
            }
            /* Capture the input info data */
            dwCoord = *pCoord;

            nMaxLength = stk->Parameters.DeviceIoControl.OutputBufferLength - sizeof(COORD);
            Irp->IoStatus.Information = 0;

            if ( dwCoord.X < 0 || dwCoord.X >= DeviceExtension->Columns ||
                 dwCoord.Y < 0 || dwCoord.Y >= DeviceExtension->Rows    ||
                 nMaxLength == 0 )
            {
                Status = STATUS_SUCCESS;
                break;
            }

            pChar = (PCHAR)(pCoord + 1);

            if (DeviceExtension->Enabled && DeviceExtension->VideoMemory)
            {
                vidmem = DeviceExtension->VideoMemory;
                offset = (dwCoord.X + dwCoord.Y * DeviceExtension->Columns) * 2;

                nMaxLength = min(nMaxLength,
                                 (DeviceExtension->Rows - dwCoord.Y)
                                    * DeviceExtension->Columns - dwCoord.X);

                for (dwCount = 0; dwCount < nMaxLength; dwCount++, pChar++)
                {
                    vidmem[offset + (dwCount * 2)] = *pChar;
                }
                Irp->IoStatus.Information = dwCount * sizeof(CHAR);
            }

            Status = STATUS_SUCCESS;
            break;
        }

        case IOCTL_CONSOLE_DRAW:
        {
            CONSOLE_DRAW ConsoleDraw;
            PCONSOLE_DRAW pConsoleDraw;
            PUCHAR Src, Dest;
            UINT32 SrcDelta, DestDelta, i;

            /* Validate output buffer */
            if (stk->Parameters.DeviceIoControl.OutputBufferLength < sizeof(CONSOLE_DRAW))
            {
                Status = STATUS_INVALID_PARAMETER;
                break;
            }
            ASSERT(Irp->MdlAddress);
            pConsoleDraw = MmGetSystemAddressForMdlSafe(Irp->MdlAddress, NormalPagePriority);
            if (pConsoleDraw == NULL)
            {
                Status = STATUS_INSUFFICIENT_RESOURCES;
                break;
            }
            /* Capture the input info data */
            ConsoleDraw = *pConsoleDraw;

            /* Check whether we have the size for the header plus the data area */
            if ((stk->Parameters.DeviceIoControl.OutputBufferLength - sizeof(CONSOLE_DRAW)) / 2
                    < ((ULONG)ConsoleDraw.SizeX * (ULONG)ConsoleDraw.SizeY))
            {
                Status = STATUS_INVALID_BUFFER_SIZE;
                break;
            }

            Irp->IoStatus.Information = 0;

            /* Set the cursor position, clipping it to the screen */
            DeviceExtension->CursorX = min(max(ConsoleDraw.CursorX, 0), DeviceExtension->Columns - 1);
            DeviceExtension->CursorY = min(max(ConsoleDraw.CursorY, 0), DeviceExtension->Rows    - 1);
            if (DeviceExtension->Enabled)
                ScrSetCursor(DeviceExtension);
