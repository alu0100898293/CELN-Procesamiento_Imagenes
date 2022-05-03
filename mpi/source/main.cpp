#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <mpi.h>
#include <SFML/Graphics/Image.hpp>
#include "MpiFunctions.hpp"


void alignChannel(int& channelValue)
{
    channelValue = (channelValue > 255) ? 255 : channelValue;
    channelValue = (channelValue < 0) ? 0 : channelValue;
}

void applyFilter(sf::Image& image)
{
    const int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1} };

    const int Gy[3][3] = {
        {-1, -2, -1},
        {0,  0, 0  },
        {1,  2, 1  } };

    //const auto kernelSize = static_cast<int>(filter.size());
    //const auto kernelMargin = kernelSize / 2;
    const auto imageHeight = static_cast<int>(image.getSize().y);
    const auto imageWidth = static_cast<int>(image.getSize().x);
    auto outputImage = image;

    const auto rowsPerProcess = imageHeight / MPI::getWorldSize();
    auto processBeginRow = MPI::getRank() * rowsPerProcess;
    auto processEndRow = MPI::getRank() * rowsPerProcess + rowsPerProcess;


    if (processBeginRow == 0) processBeginRow += 1;
    if (MPI::getRank()==MPI::getWorldSize()-1) processEndRow = imageHeight-1;

    for (int y = processBeginRow; y <processEndRow; ++y)
    {
        for (int x = 1; x < imageWidth - 1; ++x)
        {
            int newRedChannel{}, newGreenChannel{}, newBlueChannel{};
            int GxRed{}, GyRed{}, GxBlue{}, GyBlue{}, GxGreen{}, GyGreen{};
            for (int kernelX = -1; kernelX <= 1; ++kernelX)
            {
                for (int kernelY = -1; kernelY <= 1; ++kernelY)
                {
                    const auto GxValue = Gx[kernelX + 1][kernelY + 1];
                    const auto GyValue = Gy[kernelX + 1][kernelY + 1];
                    const auto pixel = image.getPixel(x + kernelX, y + kernelY);

                    //Gradient X
                    GxRed += static_cast<int>(pixel.r * GxValue);
                    GxGreen += static_cast<int>(pixel.g * GxValue);
                    GxBlue += static_cast<int>(pixel.b * GxValue);

                    //Gradient Y
                    GyRed += static_cast<int>(pixel.r * GyValue);
                    GyGreen += static_cast<int>(pixel.g * GyValue);
                    GyBlue += static_cast<int>(pixel.b * GyValue);
                }
            }

            newRedChannel = abs(GxRed) + abs(GyRed);
            newGreenChannel = abs(GxGreen) + abs(GyGreen);
            newBlueChannel = abs(GxBlue) + abs(GyBlue);
            
            alignChannel(newRedChannel);
            alignChannel(newGreenChannel);
            alignChannel(newBlueChannel);
            outputImage.setPixel(x, y, sf::Color(newRedChannel, newGreenChannel, newBlueChannel));
        }
    }

    image = std::move(outputImage);
}

sf::Image distributeImage(const sf::Image& image)
{
    sf::Image processImage;
    std::uint32_t dataSize, imageWidth, imageHeight;
    std::vector<sf::Uint8> buffer{};

    if (MPI::isMasterProcess())
    {
        dataSize = image.getSize().x * image.getSize().y * 4;
        imageWidth = image.getSize().x;
        imageHeight = image.getSize().y;
        buffer = std::vector<sf::Uint8>(image.getPixelsPtr(), image.getPixelsPtr() + dataSize);
    }

    MPI_Bcast(&dataSize, 1, MPI_UINT32_T, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(&imageWidth, 1, MPI_UINT32_T, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(&imageHeight, 1, MPI_UINT32_T, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

    buffer.resize(dataSize);
    MPI_Bcast(buffer.data(), static_cast<int>(buffer.size()), MPI_UNSIGNED_CHAR, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

    processImage.create(imageWidth, imageHeight, buffer.data());
    return processImage;
}

void collectImage(sf::Image& image, std::vector<sf::Uint8>& buffer)
{
    const auto dataSizePerProcess = static_cast<int>(buffer.size() / MPI::getWorldSize());
    MPI_Gather(image.getPixelsPtr() + MPI::getRank() * dataSizePerProcess, dataSizePerProcess, MPI_UNSIGNED_CHAR,
        buffer.data(), dataSizePerProcess, MPI_UNSIGNED_CHAR, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
}

sf::Image loadImageForMaster(std::string &imageName)
{
    sf::Image image{};
    if (MPI::isMasterProcess())
    {
        if(imageName.empty()){
            std::cout << "No image specified, using default" << std::endl;
            imageName = "hoja.jpg";
        }
            
        image.loadFromFile("../images/" + imageName);
    }
    return image;
}

void reconstructImage(sf::Image& image, const std::vector<sf::Uint8>& buffer)
{
    if (MPI::isMasterProcess())
    {
        image.create(image.getSize().x, image.getSize().y, buffer.data());
    }
}

void saveImage(sf::Image& image, std::string imageName)
{
    if (MPI::isMasterProcess())
    {
        image.saveToFile("./out/" + imageName);
    }
}

int calculateImageSize(const sf::Image& image)
{
    return image.getSize().x * image.getSize().y * 4;
}

void logDuration(const uint64_t duration)
{
    if (MPI::isMasterProcess())
    {
        std::cout << "Duration [ms]: " << duration << std::endl;
    }
}

int main(int argc, char** argv)
{
    double t1, t2;

    std::string imageName = argv[1];
    sf::Image image;

    MPI_Init(&argc, &argv);

    if (MPI::isMasterProcess())
        image = loadImageForMaster(imageName);
    
    if (MPI::isMasterProcess())
        t1 = MPI_Wtime();

    image = distributeImage(image);
 
    std::vector<sf::Uint8> buffer{};
    buffer = std::vector<sf::Uint8>(calculateImageSize(image));

    applyFilter(image);
    
    collectImage(image, buffer);

    reconstructImage(image, buffer);

    if (MPI::isMasterProcess()){
        t2 = MPI_Wtime();
        std::cout << "\tElapsed time for process " << MPI::getRank() << ": " << (t2-t1)*1000 << " ms" << std::endl;
    }

    if (MPI::isMasterProcess())
        saveImage(image, imageName);
        
    MPI_Finalize();
    
    return EXIT_SUCCESS;
}