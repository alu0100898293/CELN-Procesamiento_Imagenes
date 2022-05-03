#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <SFML/Graphics/Image.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__device__ void alignChannel(int& channelValue)
{
    channelValue = (channelValue > 255) ? 255 : channelValue;
    channelValue = (channelValue < 0) ? 0 : channelValue;
}

__global__ void applyFilterOnCuda(
    const sf::Uint8* inputImageData, sf::Uint8* outputImageData,
    const std::size_t width, const std::size_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1} };

    const int Gy[3][3] = {
        {-1, -2, -1},
        {0,  0, 0  },
        {1,  2, 1  } };

    if (((x - 1) > 0 && (x + 1) < width) &&
        ((y - 1) > 0 && (y + 1) < height))
    {
        int newRedChannel{}, newGreenChannel{}, newBlueChannel{};
        int GxRed{}, GyRed{}, GxBlue{}, GyBlue{}, GxGreen{}, GyGreen{};
        for (int kernelX = -1; kernelX <= 1; ++kernelX)
        {
            for (int kernelY = -1; kernelY <= 1; ++kernelY)
            {
                const auto GxValue = Gx[kernelX + 1][kernelY + 1];
                const auto GyValue = Gy[kernelX + 1][kernelY + 1];

                const auto pixel = &inputImageData[((x + kernelX) + (y + kernelY) * width) * 4];

                //Gradient X
                GxRed += static_cast<int>(pixel[0] * GxValue);
                GxGreen += static_cast<int>(pixel[1] * GxValue);
                GxBlue += static_cast<int>(pixel[2] * GxValue);

                //Gradient Y
                GyRed += static_cast<int>(pixel[0] * GyValue);
                GyGreen += static_cast<int>(pixel[1]* GyValue);
                GyBlue += static_cast<int>(pixel[2] * GyValue);
            }
        }

        newRedChannel = abs(GxRed) + abs(GyRed);
        newGreenChannel = abs(GxGreen) + abs(GyGreen);
        newBlueChannel = abs(GxBlue) + abs(GyBlue);
        
        alignChannel(newRedChannel);
        alignChannel(newGreenChannel);
        alignChannel(newBlueChannel);

        auto outPixel = &outputImageData[(x + y * width) * 4];
        outPixel[0] = newRedChannel;
        outPixel[1] = newGreenChannel;
        outPixel[2] = newBlueChannel;
    }
}

auto calculateImageSize(const sf::Image& image)
{
    return image.getSize().x * image.getSize().y * 4;
}

void applyFilter(sf::Image& image, int block)
{
    thrust::host_vector<sf::Uint8> hostImageData{ image.getPixelsPtr(), image.getPixelsPtr() + calculateImageSize(image) };
    thrust::device_vector<sf::Uint8> devImageData(calculateImageSize(image));
    thrust::device_vector<sf::Uint8> devOutputImageData(calculateImageSize(image));
    thrust::copy(hostImageData.begin(), hostImageData.end(), devImageData.begin());

    dim3 dimBlock(block, block);
    dim3 dimGrid(static_cast<uint32_t>(ceil((float)image.getSize().x / dimBlock.x)),
                   static_cast<uint32_t>(ceil((float)image.getSize().y / dimBlock.y)));

    applyFilterOnCuda<<<dimGrid, dimBlock>>>(
        devImageData.data().get(), devOutputImageData.data().get(),
        image.getSize().x, image.getSize().y);

    thrust::copy(devOutputImageData.begin(), devOutputImageData.end(), hostImageData.begin());
    image.create(image.getSize().x, image.getSize().y, hostImageData.data());
}


sf::Image  loadImage(std::string &imageName)
{
    sf::Image image{};
    
    if(imageName.empty()){
        std::cout << "No image specified, using default" << std::endl;
        imageName = "hoja.jpg";
    }
        
    image.loadFromFile("../images/" + imageName);
    
    return image;
}

void saveImage(sf::Image& image,std::string imageName)
{
    image.saveToFile("./out/" + imageName);
}

int main(int argc, char** argv)
{
    // Timers
    cudaEvent_t start, stop;
    
    std::string imageName;
    int block;
    /* Command line parameters processing */
    switch(argc) {
        case 3: 
                imageName = argv[1];
                block = atoi(argv[2]);
                break;
        default: 
                printf("\nUse: %s <Img_Name>  <Dim_Block>", argv[0]);
                break;
    }
    

    sf::Image image = loadImage(imageName);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    applyFilter(image, block);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float timeMs{};
    cudaEventElapsedTime(&timeMs, start, stop);
    std::cout << "Time: " << timeMs << " ms" << std::endl;

    saveImage(image, imageName);


    return EXIT_SUCCESS;
}