#ifndef __DEVICE_ABOUT_H__
#define __DEVICE_ABOUT_H__

#include <vector>
#include <string>

namespace DeviceAbout {

    struct CudaDeviceAbout {
        std::string deviceName;
        int computeCapabilityMajor;
        int computeCapabilityMinor;

        int driverVersionMajor;
        int driverVersionMinor;

        std::size_t totalGlobalMemory;
        int multiProcessorCount;
        int cudaCoreCount;
        int clockRate;
        int memoryClockRate;
        int globalMemoryBusWidth;
        int L2CacheSize;

        int maximumTexture1dWidth;
        int maximumTexture2dWidth;
        int maximumTexture2dHeight;
        int maximumTexture3dWidth;
        int maximumTexture3dHeight;
        int maximumTexture3dDepth;
        int maximumTexture1dLayeredWidth;
        int maximumTexture1dLayeredLayers;
        int maximumTexture2dLayeredWidth;
        int maximumTexture2dLayeredHeight;
        int maximumTexture2dLayeredLayers;

        int totalConstantMemory;
        int maxSharedMemoryPerBlock;
        int maxRegistersPerBlock;
        int warpSize;
        int maxThreadsPerMultiProcessor;
        int maxThreadsPerBlock;

        int maxBlockDim[3];
        int maxGridDim[3];

        int textureAlignment;
        int maxPitch;
        bool gpuOverlap;
        int asyncEngineCount;

        bool kernelExecTimeoutEnabled;
        bool integrated;
        bool canMapHostMemory;
        bool concurrentKernels;
        bool surfaceAlignment;
        bool eccEnabled;
//        bool tccDriver;
        bool unifiedAddressing;
        int pciBusID, pciDeviceID;
        std::vector<int> pearToPearAccessFrom;
        std::vector<int> pearToPearAccessTo;
    };


    class CudaDevices {
    public:
        typedef std::vector<CudaDeviceAbout>::const_iterator const_iterator;

        CudaDevices();
        const_iterator begin() const;
        const_iterator end() const;
        int deviceCount() const;
    };

} // namespace DeviceAbout


#endif //__DEVICE_ABOUT_H__
