	FailWithAction(inDriver != gAudioServerPlugInDriverRef, theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_BeginIOOperation: bad driver reference");
	FailWithAction(inDeviceObjectID != kObjectID_Device, theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_BeginIOOperation: bad device ID");

Done:
	return theAnswer;
}

static OSStatus	BlackHole_DoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, AudioObjectID inStreamObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo* inIOCycleInfo, void* ioMainBuffer, void* ioSecondaryBuffer)
{
	//	This is called to actuall perform a given operation. For this device, all we need to do is
	//	clear the buffer for the ReadInput operation.
	
	#pragma unused(inClientID, inIOCycleInfo, ioSecondaryBuffer)
	
	//	declare the local variables
	OSStatus theAnswer = 0;
	
	//	check the arguments
	FailWithAction(inDriver != gAudioServerPlugInDriverRef, theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_DoIOOperation: bad driver reference");
	FailWithAction(inDeviceObjectID != kObjectID_Device, theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_DoIOOperation: bad device ID");
	FailWithAction((inStreamObjectID != kObjectID_Stream_Input) && (inStreamObjectID != kObjectID_Stream_Output), theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_DoIOOperation: bad stream ID");
    
    
    // copy internal ring buffer to io buffer
	if(inOperationID == kAudioServerPlugInIOOperationReadInput)
	{
        // calculate the ring buffer offset for the first sample INPUT
        ringBufferOffset = ((UInt64)(inIOCycleInfo->mInputTime.mSampleTime * BYTES_PER_FRAME) % RING_BUFFER_SIZE);
        
        // calculate the size of the buffer
        inIOBufferByteSize = inIOBufferFrameSize * BYTES_PER_FRAME;
        remainingRingBufferByteSize = RING_BUFFER_SIZE - ringBufferOffset;
        
        if (remainingRingBufferByteSize > inIOBufferByteSize)
        {

            // copy whole buffer if we have space
            memcpy(ioMainBuffer, ringBuffer + ringBufferOffset, inIOBufferByteSize);

            // clear the internal ring buffer
            memset(ringBuffer + ringBufferOffset, 0, inIOBufferByteSize);

        }
        else
        {

            // copy 1st half
            memcpy(ioMainBuffer, ringBuffer + ringBufferOffset, remainingRingBufferByteSize);
            // copy 2nd half
            memcpy(ioMainBuffer + remainingRingBufferByteSize, ringBuffer, inIOBufferByteSize - remainingRingBufferByteSize);

            // clear the 1st half
            memset(ringBuffer + ringBufferOffset, 0, remainingRingBufferByteSize);
            // clear the 2nd half
            memset(ringBuffer, 0, inIOBufferByteSize - remainingRingBufferByteSize);
        }
    }
    
    // copy io buffer to internal ring buffer
    if(inOperationID == kAudioServerPlugInIOOperationWriteMix)
    {
        // TODO Mix inputs instead of over writing. 
        
        // calculate the ring buffer offset for the first sample OUTPUT
        ringBufferOffset = ((UInt64)(inIOCycleInfo->mOutputTime.mSampleTime * BYTES_PER_FRAME) % RING_BUFFER_SIZE);
        
        // calculate the size of the buffer
        inIOBufferByteSize = inIOBufferFrameSize * BYTES_PER_FRAME;

        // mix the audio
        for(UInt64 sample = 0; sample < inIOBufferByteSize; sample += sizeof(Float32))
        {
            // sample from ioMainBuffer
            Float32* ioSample = ioMainBuffer + sample;
                
            // sample from ring buffer
            Float32* ringSample = (Float32*)(ringBuffer + (ringBufferOffset + sample) % RING_BUFFER_SIZE);
            
            // mix the two together
            *ringSample += *ioSample;
        }
        
