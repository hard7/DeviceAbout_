#include "DeviceAbout/DeviceAbout.h"
#include <iostream>
#include <algorithm>

using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    cout << DeviceAbout::CudaDevices::deviceCount() << endl;

    std::vector<std::size_t> memories;

    using DeviceAbout::CudaDevices;
    using DeviceAbout::CudaDeviceAbout;

    std::transform(CudaDevices::begin(), CudaDevices::end(), std::back_inserter(memories), [](CudaDeviceAbout const& about) {
        return about.totalGlobalMemory;
    });



    for(DeviceAbout::CudaDeviceAbout const& about : DeviceAbout::CudaDevices()) {
        cout << about.totalGlobalMemory / (1024 * 1024 * 1024.) << endl;
    }


    return 0;
}