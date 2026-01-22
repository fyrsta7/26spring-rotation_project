	FailWithAction(inDriver != gAudioServerPlugInDriverRef, theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_BeginIOOperation: bad driver reference");
	FailWithAction(inDeviceObjectID != kObjectID_Device, theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_BeginIOOperation: bad device ID");

Done:
	return theAnswer;
}

static OSStatus	BlackHole_DoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, AudioObjectID inStreamObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo* inIOCycleInfo, void* ioMainBuffer, void* ioSecondaryBuffer)
{
	//	This is called to actually perform a given operation. 
	
	#pragma unused(inClientID, inIOCycleInfo, ioSecondaryBuffer)
	
	//	declare the local variables
	OSStatus theAnswer = 0;
	
	//	check the arguments
	FailWithAction(inDriver != gAudioServerPlugInDriverRef, theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_DoIOOperation: bad driver reference");
	FailWithAction(inDeviceObjectID != kObjectID_Device, theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_DoIOOperation: bad device ID");
	FailWithAction((inStreamObjectID != kObjectID_Stream_Input) && (inStreamObjectID != kObjectID_Stream_Output), theAnswer = kAudioHardwareBadObjectError, Done, "BlackHole_DoIOOperation: bad stream ID");

    // Calculate the ring buffer offsets and splits.
    UInt64 mSampleTime = inOperationID == kAudioServerPlugInIOOperationReadInput ? inIOCycleInfo->mInputTime.mSampleTime : inIOCycleInfo->mOutputTime.mSampleTime;
    UInt32 ringBufferFrameLocationStart = mSampleTime % kRing_Buffer_Frame_Size;
    UInt32 firstPartFrameSize = kRing_Buffer_Frame_Size - ringBufferFrameLocationStart;
    UInt32 secondPartFrameSize = 0;
    
    if (firstPartFrameSize >= inIOBufferFrameSize)
    {
        firstPartFrameSize = inIOBufferFrameSize;
    }
    else
    {
        secondPartFrameSize = inIOBufferFrameSize - firstPartFrameSize;
    }
    
    // Keep track of last outputSampleTime and the cleared buffer status.
    static Float64 lastOutputSampleTime = 0;
    static Boolean isBufferClear = true;
    
    // From BlackHole to Application
    if(inOperationID == kAudioServerPlugInIOOperationReadInput)
    {
        // If mute is one let's just fill the buffer with zeros or if there's no apps outputing audio
        if (gMute_Master_Value || lastOutputSampleTime - inIOBufferFrameSize < inIOCycleInfo->mInputTime.mSampleTime)
        {
            // Clear the ioMainBuffer
            vDSP_vclr(ioMainBuffer, 1, inIOBufferFrameSize * kNumber_Of_Channels);
            
            // Clear the ring buffer.
            if (!isBufferClear)
            {
                vDSP_vclr(gRingBuffer, 1, kRing_Buffer_Frame_Size * kNumber_Of_Channels);
                isBufferClear = true;
            }
        }
        else
        {
            // Copy the buffers.
            cblas_scopy(firstPartFrameSize * kNumber_Of_Channels, gRingBuffer + ringBufferFrameLocationStart * kNumber_Of_Channels, 1, ioMainBuffer, 1);
            cblas_scopy(secondPartFrameSize * kNumber_Of_Channels, gRingBuffer, 1, (Float32*)ioMainBuffer + firstPartFrameSize * kNumber_Of_Channels, 1);

            // Finally we'll apply the output volume to the buffer.
            vDSP_vsmul(ioMainBuffer, 1, &gVolume_Master_Value, ioMainBuffer, 1, inIOBufferFrameSize * kNumber_Of_Channels);
        }
        

    }
    
    // From Application to BlackHole
    if(inOperationID == kAudioServerPlugInIOOperationWriteMix)
    {
        // Save the last output time.
        lastOutputSampleTime= inIOCycleInfo->mOutputTime.mSampleTime;
        isBufferClear = false;
        
        // Copy the buffers.
        cblas_scopy(firstPartFrameSize * kNumber_Of_Channels, ioMainBuffer, 1, gRingBuffer + ringBufferFrameLocationStart * kNumber_Of_Channels, 1);
