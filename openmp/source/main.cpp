#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <omp.h>
#include <SFML/Graphics/Image.hpp>


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

    const auto imageHeight = static_cast<int>(image.getSize().y);
    const auto imageWidth = static_cast<int>(image.getSize().x);
    auto outputImage = image;
    
    #pragma omp parallel for
    for (int x = 1; x < imageWidth - 1; ++x)
    {
        for (int y = 1; y < imageHeight - 1; ++y)
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

sf::Image loadImage(std::string &imageName)
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
    std::string imageName = argv[1];

    sf::Image image = loadImage(imageName);

    std::cout << "Filtering image"<< std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    applyFilter(image);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "\tElapsed time: " << (float) (duration) << " ms" << std::endl;

    saveImage(image, imageName);

    return EXIT_SUCCESS;
}