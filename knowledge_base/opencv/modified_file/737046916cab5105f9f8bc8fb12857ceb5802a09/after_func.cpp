                ++_pR;
            }else
            {
                if (colorIndex<21){
                    colorIndex=1;
                    ++_pG;
                }else{
                    colorIndex=2;
                    ++_pB;
                }
            }
            _colorSampling[index] = colorIndex*this->getNBpixels()+index;
        }
        _pR/=(float)this->getNBpixels();
        _pG/=(float)this->getNBpixels();
        _pB/=(float)this->getNBpixels();
        std::cout<<"Color channels proportions: pR, pG, pB= "<<_pR<<", "<<_pG<<", "<<_pB<<", "<<std::endl;
        break;
    case RETINA_COLOR_DIAGONAL:
        for (unsigned int index=0 ; index<this->getNBpixels(); ++index)
        {
            _colorSampling[index] = index+((index%3+(index%_filterOutput.getNBcolumns()))%3)*_filterOutput.getNBpixels();
        }
        _pR=_pB=_pG=1.f/3;
        break;
    case RETINA_COLOR_BAYER: // default sets bayer sampling
        for (unsigned int index=0 ; index<_filterOutput.getNBpixels(); ++index)
        {
            //First line: R G R G
            _colorSampling[index] = index+((index/_filterOutput.getNBcolumns())%2)*_filterOutput.getNBpixels()+((index%_filterOutput.getNBcolumns())%2)*_filterOutput.getNBpixels();
            //First line: G R G R
            //_colorSampling[index] = 3*index+((index/_filterOutput.getNBcolumns())%2)+((index%_filterOutput.getNBcolumns()+1)%2);
        }
        _pR=_pB=0.25;
        _pG=0.5;
        break;
    default:
#ifdef RETINACOLORDEBUG
        std::cerr<<"RetinaColor::No or wrong color sampling method, skeeping"<<std::endl;
#endif
        return;
        break;//.. not useful, yes

    }
    // feeling the mosaic buffer:
    _RGBmosaic=0;
    for (unsigned int index=0 ; index<_filterOutput.getNBpixels(); ++index)
        // the RGB _RGBmosaic buffer contains 1 where the pixel corresponds to a sampled color
        _RGBmosaic[_colorSampling[index]]=1.0;

    // computing photoreceptors local density
    _spatiotemporalLPfilter(&_RGBmosaic[0], &_colorLocalDensity[0]);
    _spatiotemporalLPfilter(&_RGBmosaic[0]+_filterOutput.getNBpixels(), &_colorLocalDensity[0]+_filterOutput.getNBpixels());
    _spatiotemporalLPfilter(&_RGBmosaic[0]+_filterOutput.getDoubleNBpixels(), &_colorLocalDensity[0]+_filterOutput.getDoubleNBpixels());
    unsigned int maxNBpixels=3*_filterOutput.getNBpixels();
    register float *colorLocalDensityPTR=&_colorLocalDensity[0];
    for (unsigned int i=0;i<maxNBpixels;++i, ++colorLocalDensityPTR)
        *colorLocalDensityPTR=1.f/ *colorLocalDensityPTR;

#ifdef RETINACOLORDEBUG
    std::cout<<"INIT    _colorLocalDensity max, min: "<<_colorLocalDensity.max()<<", "<<_colorLocalDensity.min()<<std::endl;
#endif
    // end of the init step
    _objectInit=true;
}

// public functions

void RetinaColor::runColorDemultiplexing(const std::valarray<float> &multiplexedColorFrame, const bool adaptiveFiltering, const float maxInputValue)
{
    // demultiplex the grey frame to RGB frame
    // -> first set demultiplexed frame to 0
    _demultiplexedTempBuffer=0;
    // -> demultiplex process
    register unsigned int *colorSamplingPRT=&_colorSampling[0];
    register const float *multiplexedColorFramePtr=get_data(multiplexedColorFrame);
    for (unsigned int indexa=0; indexa<_filterOutput.getNBpixels() ; ++indexa)
        _demultiplexedTempBuffer[*(colorSamplingPRT++)]=*(multiplexedColorFramePtr++);

    // interpolate the demultiplexed frame depending on the color sampling method
    if (!adaptiveFiltering)
        _interpolateImageDemultiplexedImage(&_demultiplexedTempBuffer[0]);

    // low pass filtering the demultiplexed frame
    _spatiotemporalLPfilter(&_demultiplexedTempBuffer[0], &_chrominance[0]);
    _spatiotemporalLPfilter(&_demultiplexedTempBuffer[0]+_filterOutput.getNBpixels(), &_chrominance[0]+_filterOutput.getNBpixels());
    _spatiotemporalLPfilter(&_demultiplexedTempBuffer[0]+_filterOutput.getDoubleNBpixels(), &_chrominance[0]+_filterOutput.getDoubleNBpixels());

    /*if (_samplingMethod=BAYER)
    {
        _applyRIFfilter(_chrominance, _chrominance);
        _applyRIFfilter(_chrominance+_filterOutput.getNBpixels(), _chrominance+_filterOutput.getNBpixels());
        _applyRIFfilter(_chrominance+_filterOutput.getDoubleNBpixels(), _chrominance+_filterOutput.getDoubleNBpixels());
    }*/

    // normalize by the photoreceptors local density and retrieve the local luminance
    register float *chrominancePTR= &_chrominance[0];
    register float *colorLocalDensityPTR= &_colorLocalDensity[0];
    register float *luminance= &(*_luminance)[0];
    if (!adaptiveFiltering)// compute the gradient on the luminance
    {
        if (_samplingMethod==RETINA_COLOR_RANDOM)
            for (unsigned int indexc=0; indexc<_filterOutput.getNBpixels() ; ++indexc, ++chrominancePTR, ++colorLocalDensityPTR, ++luminance)
            {
                // normalize by photoreceptors density
                float Cr=*(chrominancePTR)*_colorLocalDensity[indexc];
                float Cg=*(chrominancePTR+_filterOutput.getNBpixels())*_colorLocalDensity[indexc+_filterOutput.getNBpixels()];
                float Cb=*(chrominancePTR+_filterOutput.getDoubleNBpixels())*_colorLocalDensity[indexc+_filterOutput.getDoubleNBpixels()];
                *luminance=(Cr+Cg+Cb)*_pG;
                *(chrominancePTR)=Cr-*luminance;
                *(chrominancePTR+_filterOutput.getNBpixels())=Cg-*luminance;
                *(chrominancePTR+_filterOutput.getDoubleNBpixels())=Cb-*luminance;
            }
        else
            for (unsigned int indexc=0; indexc<_filterOutput.getNBpixels() ; ++indexc, ++chrominancePTR, ++colorLocalDensityPTR, ++luminance)
            {
                float Cr=*(chrominancePTR);
                float Cg=*(chrominancePTR+_filterOutput.getNBpixels());
                float Cb=*(chrominancePTR+_filterOutput.getDoubleNBpixels());
                *luminance=_pR*Cr+_pG*Cg+_pB*Cb;
                *(chrominancePTR)=Cr-*luminance;
                *(chrominancePTR+_filterOutput.getNBpixels())=Cg-*luminance;
                *(chrominancePTR+_filterOutput.getDoubleNBpixels())=Cb-*luminance;
            }

        // in order to get the color image, each colored map needs to be added the luminance
        // -> to do so, compute:  multiplexedColorFrame - remultiplexed chrominances
        runColorMultiplexing(_chrominance, _tempMultiplexedFrame);
        //lum = 1/3((f*(ImR))/(f*mR) + (f*(ImG))/(f*mG) + (f*(ImB))/(f*mB));
        float *luminancePTR= &(*_luminance)[0];
        chrominancePTR= &_chrominance[0];
        float *demultiplexedColorFramePTR= &_demultiplexedColorFrame[0];
        for (unsigned int indexp=0; indexp<_filterOutput.getNBpixels() ; ++indexp, ++luminancePTR, ++chrominancePTR, ++demultiplexedColorFramePTR)
        {
            *luminancePTR=(multiplexedColorFrame[indexp]-_tempMultiplexedFrame[indexp]);
            *(demultiplexedColorFramePTR)=*(chrominancePTR)+*luminancePTR;
            *(demultiplexedColorFramePTR+_filterOutput.getNBpixels())=*(chrominancePTR+_filterOutput.getNBpixels())+*luminancePTR;
            *(demultiplexedColorFramePTR+_filterOutput.getDoubleNBpixels())=*(chrominancePTR+_filterOutput.getDoubleNBpixels())+*luminancePTR;
        }

    }else
    {
        register const float *multiplexedColorFramePTR= get_data(multiplexedColorFrame);
        for (unsigned int indexc=0; indexc<_filterOutput.getNBpixels() ; ++indexc, ++chrominancePTR, ++colorLocalDensityPTR, ++luminance, ++multiplexedColorFramePTR)
