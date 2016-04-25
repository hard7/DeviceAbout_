#include "../DeviceAbout.h"

#include <cstdio>
#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <drvapi_error_string.h>


namespace {

    DeviceAbout::CudaDeviceAbout getCudaDeviceAbout(int deviceId) {
//        int major = 0, minor = 0;
//        int deviceCount = 0;
        DeviceAbout::CudaDeviceAbout about;

        CUresult error_id;
        error_id = cuDeviceComputeCapability(&about.computeCapabilityMajor, &about.computeCapabilityMinor, deviceId);

        if(error_id != CUDA_SUCCESS) {
            printf("cuDeviceComputeCapability returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
            exit(EXIT_FAILURE);
        }

        char* deviceName = new char[256];
        error_id = cuDeviceGetName(deviceName, 256, deviceId);
        about.deviceName = std::string(std::move(deviceName));

        if (error_id != CUDA_SUCCESS) {
            printf("cuDeviceGetName returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
            exit(EXIT_FAILURE);
        }

        int driverVersion = 0;
        cuDriverGetVersion(&driverVersion);
        about.driverVersionMajor = driverVersion/1000;
        about.driverVersionMinor = (driverVersion%100)/10;


        error_id = cuDeviceTotalMem(&about.totalGlobalMemory, deviceId);
        if (error_id != CUDA_SUCCESS) {
            printf("cuDeviceTotalMem returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
            exit(EXIT_FAILURE);
        }

        getCudaAttribute<int>(&about.multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceId);
        about.cudaCoreCount = about.multiProcessorCount * _ConvertSMVer2CoresDRV(about.computeCapabilityMajor, about.computeCapabilityMinor);

        getCudaAttribute<int>(&about.clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, deviceId);
        getCudaAttribute<int>(&about.memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, deviceId);
        getCudaAttribute<int>(&about.globalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, deviceId);
        getCudaAttribute<int>(&about.L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, deviceId);

        getCudaAttribute<int>(&about.maximumTexture1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, deviceId);
        getCudaAttribute<int>(&about.maximumTexture2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, deviceId);
        getCudaAttribute<int>(&about.maximumTexture2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, deviceId);
        getCudaAttribute<int>(&about.maximumTexture3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, deviceId);
        getCudaAttribute<int>(&about.maximumTexture3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, deviceId);
        getCudaAttribute<int>(&about.maximumTexture3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, deviceId);

        getCudaAttribute<int>(&about.maximumTexture1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, deviceId);
        getCudaAttribute<int>(&about.maximumTexture1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, deviceId);

        getCudaAttribute<int>(&about.maximumTexture2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, deviceId);
        getCudaAttribute<int>(&about.maximumTexture2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, deviceId);
        getCudaAttribute<int>(&about.maximumTexture1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, deviceId);

        getCudaAttribute<int>(&about.totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, deviceId);
        getCudaAttribute<int>(&about.maxSharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, deviceId);
        getCudaAttribute<int>(&about.maxRegistersPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, deviceId);
        getCudaAttribute<int>(&about.warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, deviceId);
        getCudaAttribute<int>(&about.maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, deviceId);
        getCudaAttribute<int>(&about.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, deviceId);

        getCudaAttribute<int>(&about.maxBlockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, deviceId);
        getCudaAttribute<int>(&about.maxBlockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, deviceId);
        getCudaAttribute<int>(&about.maxBlockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, deviceId);

        getCudaAttribute<int>(&about.maxGridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, deviceId);
        getCudaAttribute<int>(&about.maxGridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, deviceId);
        getCudaAttribute<int>(&about.maxGridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, deviceId);

        getCudaAttribute<int>(&about.textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, deviceId);
        getCudaAttribute<int>(&about.maxPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, deviceId);

        int gpuOverlap;
        getCudaAttribute<int>(&gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, deviceId);
        about.gpuOverlap = static_cast<bool>(gpuOverlap);

        getCudaAttribute<int>(&about.asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, deviceId);

        int kernelExecTimeoutEnabled;
        getCudaAttribute<int>(&kernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, deviceId);
        about.kernelExecTimeoutEnabled = static_cast<bool>(kernelExecTimeoutEnabled);

        int integrated;
        getCudaAttribute<int>(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, deviceId);
        about.integrated = static_cast<bool>(integrated);

        int canMapHostMemory;
        getCudaAttribute<int>(&canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, deviceId);
        about.canMapHostMemory = static_cast<bool>(canMapHostMemory);

        int concurrentKernels;
        getCudaAttribute<int>(&concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, deviceId);
        about.concurrentKernels = static_cast<bool>(concurrentKernels);

        int surfaceAlignment;
        getCudaAttribute<int>(&surfaceAlignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, deviceId);
        about.surfaceAlignment = static_cast<bool>(surfaceAlignment);

        int eccEnabled;
        getCudaAttribute<int>(&eccEnabled,  CU_DEVICE_ATTRIBUTE_ECC_ENABLED, deviceId);
        about.eccEnabled = static_cast<bool>(eccEnabled);

//#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
//        int tccDriver ;
//    getCudaAttribute<int>(&tccDriver ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, deviceId);
//    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
//#endif

        int unifiedAddressing;
        getCudaAttribute<int>(&unifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, deviceId);
        about.unifiedAddressing = static_cast<bool>(unifiedAddressing);

        getCudaAttribute<int>(&about.pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, deviceId);
        getCudaAttribute<int>(&about.pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, deviceId);

        return about;
}

    std::vector<DeviceAbout::CudaDeviceAbout> createCudaDeviceAboutCollection() {
        CUresult error_id = cuInit(0);
        if (error_id != CUDA_SUCCESS) {
            printf("cuInit(0) returned %d\n-> %s\n", error_id, getCudaDrvErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        std::vector<DeviceAbout::CudaDeviceAbout> result;
        result.reserve(DeviceAbout::CudaDevices::deviceCount());
        for(int i=0; i<DeviceAbout::CudaDevices::deviceCount(); ++i) result.push_back(getCudaDeviceAbout(i));
        return result;
    }

    std::vector<DeviceAbout::CudaDeviceAbout> cudaDeviceAboutCollection = createCudaDeviceAboutCollection();
}

/* static */
DeviceAbout::CudaDevices::const_iterator DeviceAbout::CudaDevices::begin() {
    return cudaDeviceAboutCollection.begin();
}

/* static */
DeviceAbout::CudaDevices::const_iterator DeviceAbout::CudaDevices::end() {
    return cudaDeviceAboutCollection.end();
}

/* static */
int DeviceAbout::CudaDevices::deviceCount() {
    int deviceCount = 0;
    CUresult error_id = cuDeviceGetCount(&deviceCount);
    if (error_id != CUDA_SUCCESS) {
        printf("cuDeviceGetCount returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    return deviceCount;
}