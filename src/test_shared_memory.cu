#include <cuda_runtime.h>
#include <iostream>

int main() {
    int dev = 0; // 默认用 GPU 0
    cudaSetDevice(dev);

    int maxDefault = 0;
    int maxOptin = 0;

    cudaDeviceGetAttribute(&maxDefault, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    std::cout << "设备 ID: " << dev << std::endl;
    std::cout << "默认共享内存上限 per block: " << maxDefault << " bytes" << std::endl;
    std::cout << "Opt-in 共享内存上限 per block: " << maxOptin << " bytes" << std::endl;

    return 0;
}
