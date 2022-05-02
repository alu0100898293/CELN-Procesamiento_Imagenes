# Detección de bordes con programación paralela en C++
Este repositorio cuenta con diversos programas que ejecutan sobre una imagen a color el filtro Sobel para la detección de bordes en C++. Los programas se han divido según su modo de eejcución, ya sea secuencial o según el tipo de paralelización implementada.
## Requisitos
A continuación se describen los programas necesarios para la ejecución de los programas del repositorio, así como la manera de instalarlos en una distribución Linux:
- Cmake: sudo apt-get install cmake
- SFML: sudo apt-get install libsfml-dev
- MPI: sudo apt install mpich
## Ejecución
Para la ejecución del programa debe seguir los siguientes pasos:
1. Situarse en la carpeta del código a ejecutar.
2. Ejecutar el comando **cmake .** en dicho directorio.
3. Una vez finalizado el paso anterior, ejecutar el comando **make**.
4. Por último, debemos tener generado un ejecutable con el nombre **main**, que según el progama que usemos varía el funcionamiento:
   - Secuencial y openmp: solo es necesario ejecutar **main** seguido del nombre de la imagen.
   - MPI: se debe ejecutar un **mpi run** del ejecutable **main**, seguido de los argumentos propios de mpi como el número de procesadores y el nombre de la imagen a analizar.
   - Cuda: solo es necesario ejecutar **main** seguido del nombre de la imagen y del número de bloques por malla. La cantidad de mallas se basa en el tamaño de la imagen.
5.  **IMPORTANTE**: el programa analiza las imágenes que se localizan en el directorio **images**, así que si quiere analizar una imagen nueva a las suministradas en este repositorio, sitúela en el directorio mencionado. La imagen resultante se creará en un directorio llamado **out** dentro del directorio del programa ejecutado.