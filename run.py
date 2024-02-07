import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import random


def energy_fun(image_pom):
    distances = np.zeros_like(image_pom, dtype=np.uint8)
    distances[:, :-1, :] += np.abs(image_pom[:, :-1, :] - image_pom[:, 1:, :])
    distances[:-1, :, :] += np.abs(image_pom[:-1, :, :] - image_pom[1:, :, :])
    return distances.sum()


def right_swap_fun(x, y):
    return [
        [x, y - 1],  # left
        [x - 1, y - 1],  # up
        [x - 1, y],  # up
        [x - 1, y + 1],  # up
        [x - 1, y + 2],  # up
        [x, y + 2],  # right
        [x + 1, y - 1],  # bottom
        [x + 1, y],  # bottom
        [x + 1, y + 1],  # bottom
        [x + 1, y + 2],  # bottom
        [x, y],
        [x, y + 1]
    ]


def down_swap_fun(x, y):
    return [
        [x - 1, y - 1],
        [x - 1, y],
        [x - 1, y + 1],
        [x, y + 1],
        [x + 1, y + 1],
        [x + 2, y + 1],
        [x + 2, y],
        [x + 2, y - 1],
        [x + 1, y - 1],
        [x, y - 1],
        [x, y],
        [x + 1, y]
    ]


def generate_random(n, width):
    random_coordinates = []
    n_s = []
    for i in range(n):
        coordinate = [random.randint(0, width - 1), random.randint(0, width - 1), 0, 0]
        if coordinate[0] == width - 1:
            coordinate[2] = coordinate[0]
            coordinate[3] = coordinate[1] + 1
            random_coordinates += coordinate
            n_s += [right_swap_fun(coordinate[0], coordinate[1])]
            continue
        if coordinate[1] == width - 1:
            coordinate[2] = coordinate[0] + 1
            coordinate[3] = coordinate[1]
            random_coordinates += coordinate
            n_s += [down_swap_fun(coordinate[0], coordinate[1])]
            continue
        r = random.randint(0, 1)
        if r == 1:
            coordinate[2] = coordinate[0]
            coordinate[3] = coordinate[1] + 1
            random_coordinates += coordinate
            n_s += [right_swap_fun(coordinate[0], coordinate[1])]
        else:
            coordinate[2] = coordinate[0] + 1
            coordinate[3] = coordinate[1]
            random_coordinates += coordinate
            n_s += [down_swap_fun(coordinate[0], coordinate[1])]

    return random_coordinates, n_s


mod = SourceModule("""
    #include <math.h>

    __global__ void copy_image_to_shared_memory(unsigned char *image, unsigned char *shared_image, 
    int* energy, int width, int height) {

        long idx = threadIdx.y * width + threadIdx.x;
        long i  = idx * 3;
        shared_image[i] = image[i];
        shared_image[i+1] = image[i+1];
        shared_image[i+2] = image[i+2];

        __syncthreads();

        int right = 0;
        int down = 0;

        // Check if indices are within bounds

       if ( threadIdx.x < width - 1 ) {
            right = abs(shared_image[i] - shared_image[i + 3]) + 
            abs(shared_image[i + 1] - shared_image[i + 4]) + 
            abs(shared_image[i + 2] - shared_image[i + 5]);
       }


        if( threadIdx.y < width - 1 ) {
            down = abs(shared_image[i] - shared_image[i + 3*width]) + 
            abs(shared_image[i + 1] - shared_image[i + 1 + 3*width]) + 
            abs(shared_image[i + 2] - shared_image[i + 2 + 3*width]);
        }

        atomicAdd(energy, right);  // Use atomicAdd to safely increment energy
        atomicAdd(energy, down); 
    }

    __global__ void swap(unsigned char *shared_image, char *random, int width, 
     char *x_y, float temp, float* random_values, int energy){

        __shared__ int sharedFinalEnergy[1024];
        __shared__ unsigned char *share_image_block;
        share_image_block = shared_image;

        printf("%d", share_image_block[1]);

        long idx = (threadIdx.y * 12 + threadIdx.x);
        // printf("%d ", idx);

        long x = random[idx*2];
        long y = random[idx*2+1];

        int energyNew = 0;
        int energySwitched = 0;

        int i = x*width*3 + y*3;
        int iP = x_y[threadIdx.y * 4]*width*3 + x_y[threadIdx.y * 4+1]*3;
        int iP1 = x_y[threadIdx.y * 4+2]*width*3 + x_y[threadIdx.y * 4+3]*3;

        if( !(x < 0 || y < 0 || x >= width || y>= width) ){

            if(y < width - 1) {
                energyNew += abs(shared_image[i] - shared_image[i + 3]) + 
                abs(shared_image[i + 1] - shared_image[i + 4]) + 
                abs(shared_image[i + 2] - shared_image[i + 5]);

                if(i == iP && i + 3 ==iP1){
                    energySwitched += abs(shared_image[iP1] - shared_image[iP]) + 
                    abs(shared_image[iP1 + 1] - shared_image[iP+1]) + 
                    abs(shared_image[iP1 + 2] - shared_image[iP+2]);
                }
                else if(i + 3 == iP){
                    energySwitched += abs(shared_image[i] - shared_image[iP1]) + 
                    abs(shared_image[i + 1] - shared_image[iP1 + 1]) + 
                    abs(shared_image[i + 2] - shared_image[iP1 + 2]);
                }
                else if(i == iP){
                    energySwitched += abs(shared_image[iP1] - shared_image[i + 3]) + 
                    abs(shared_image[iP1 + 1] - shared_image[i + 4]) + 
                    abs(shared_image[iP1 + 2] - shared_image[i + 5]);
                }
                else if(i == iP1){
                    energySwitched += abs(shared_image[iP] - shared_image[i + 3]) + 
                    abs(shared_image[iP + 1] - shared_image[i + 4]) + 
                    abs(shared_image[iP + 2] - shared_image[i + 5]);
                }
                else if(i + 3 == iP1){
                    energySwitched += abs(shared_image[i] - shared_image[iP]) + 
                    abs(shared_image[i + 1] - shared_image[iP+1]) + 
                    abs(shared_image[i + 2] - shared_image[iP+2]);
                }else{
                    energySwitched += abs(shared_image[i] - shared_image[i + 3]) + 
                    abs(shared_image[i + 1] - shared_image[i + 4]) + 
                    abs(shared_image[i + 2] - shared_image[i + 5]);
                }
            }

            if(x < width - 1){
                energyNew += abs(shared_image[i] - shared_image[i + 3*width]) + 
                abs(shared_image[i + 1] - shared_image[i + 1 + 3*width]) + 
                abs(shared_image[i + 2] - shared_image[i + 2 + 3*width]);

                if(i == iP && i + 3*width == iP1){
                    energySwitched +=  abs(shared_image[iP1] - shared_image[iP]) + 
                    abs(shared_image[iP1 + 1] - shared_image[iP + 1]) + 
                    abs(shared_image[iP1 + 2] - shared_image[iP + 2]); 
                }
                else if(i + 3*width == iP){
                   energySwitched +=  abs(shared_image[i] - shared_image[iP1]) + 
                   abs(shared_image[i + 1] - shared_image[iP1 + 1]) + 
                   abs(shared_image[i + 2] - shared_image[iP1 + 2]); 
                }
                else if(i == iP){
                    energySwitched += abs(shared_image[iP1] - shared_image[i + 3*width]) + 
                    abs(shared_image[iP1+1] - shared_image[i + 1 + 3*width]) + 
                    abs(shared_image[iP1+2] - shared_image[i + 2 + 3*width]);
                }
                else if(i + 3*width == iP1){
                    energySwitched +=  abs(shared_image[i] - shared_image[iP]) + 
                    abs(shared_image[i + 1] - shared_image[iP + 1]) + 
                    abs(shared_image[i + 2] - shared_image[iP + 2]); 
                }
                else if(i == iP1){
                    energySwitched += abs(shared_image[iP] - shared_image[i + 3*width]) + 
                    abs(shared_image[iP+1] - shared_image[i + 1 + 3*width]) + 
                    abs(shared_image[iP+2] - shared_image[i + 2 + 3*width]);
                }else{
                    energySwitched += abs(shared_image[i] - shared_image[i + 3*width]) + 
                    abs(shared_image[i + 1] - shared_image[i + 1 + 3*width]) + 
                    abs(shared_image[i + 2] - shared_image[i + 2 + 3*width]);
                }
            }

            // results -> (newEnergy, 

        }

        //shared

        atomicAdd(&sharedFinalEnergy[threadIdx.y*4 ], energyNew);  
        atomicAdd(&sharedFinalEnergy[threadIdx.y*4 + 1], energySwitched);  
        sharedFinalEnergy[threadIdx.y*4 + 2] = iP;  
        sharedFinalEnergy[threadIdx.y*4 + 3] = iP1;

       __syncthreads(); 

       //printf(" x: %d y: %d Energy: %d Energy switched: %d i: %d ip: %d ip1: %d idx: %d |", x, y, energyNew, energySwitched, i, 
        //iP, iP1, idx);

        if(threadIdx.x == 0 && threadIdx.y == 0){
            int bestLowEnergy = 5000000;
            int i1G = 0;
            int i2G = 0;
            for (int i = 0; i < 32; i+=4) {
                int energyVal = sharedFinalEnergy[i+1] - sharedFinalEnergy[i];
                if(energyVal < bestLowEnergy){
                    bestLowEnergy = energyVal;
                    i1G = sharedFinalEnergy[i+2];
                    i2G = sharedFinalEnergy[i+3];
                }
            }
            // 20 22    20 - 22 = -2
            __syncthreads(); 

            printf("%f ", random_values[i1G%12]);
            printf("%f", exp2f(-bestLowEnergy / temp));
            if(bestLowEnergy < 0 || random_values[i1G%12] < exp2f(-bestLowEnergy / temp)) {

            }

            printf("|%d %d %d %f|", bestLowEnergy, i1G, i2G, temp);
        }




        // if idx = 0 {... matrix = [10101010]} energija 1 x1,y1 x2,y2

    }


""")


image = np.random.randint(0, 2, (5, 5, 3), dtype=np.uint8)
shared_image = np.zeros((5, 5, 3), dtype=np.uint8)
energy = np.array([0], dtype=np.int32)  # Use an array for energy to pass it by reference

copy_image_to_shared_memory = mod.get_function("copy_image_to_shared_memory")
image_gpu = cuda.to_device(image)
shared_image_gpu = cuda.to_device(shared_image)
energy_gpu = cuda.to_device(energy)

block_size = (5, 5, 1)
grid_size = (1, 1, 1)
copy_image_to_shared_memory(image_gpu, shared_image_gpu, energy_gpu, np.int32(image.shape[1]),
                            block=block_size, grid=grid_size)

cuda.memcpy_dtoh(shared_image, shared_image_gpu)
cuda.memcpy_dtoh(energy, energy_gpu)
print(shared_image)
print(energy)

swap = mod.get_function("swap")
random_c, neighbor = generate_random(8, 5)

random_np = np.array(neighbor, dtype=np.int8)
random_xy_np = np.array(random_c, dtype=np.int8)
random_xy_gpy = cuda.to_device(random_xy_np)
random_gpu = cuda.to_device(random_np)

Ts = 1000
imax = 30000000
t = 8000 / 10000
temp = (1 - t) * 100

random_uniform_values = np.random.rand(12).astype(np.float32)
random_uniform_values_gpu = cuda.to_device(random_uniform_values)
print(random_uniform_values)

swap(image_gpu, random_gpu, np.int32(image.shape[1]), random_xy_gpy, np.float32(temp), random_uniform_values_gpu,
     np.int32(energy), block=(12, 8, 1), grid=(5, 5, 1))

