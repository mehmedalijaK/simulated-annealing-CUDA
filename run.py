import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def energy_fun(image):
    distances = np.zeros_like(image, dtype=np.uint8)
    distances[:, :-1, :] += np.abs(image[:, :-1, :] - image[:, 1:, :])
    distances[:-1, :, :] += np.abs(image[:-1, :, :] - image[1:, :, :])
    return distances.sum()


mod = SourceModule("""
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
     char *x_y){
        int idx = threadIdx.x * 2;
        long x = random[idx];
        long y = random[idx+1];
        
        
        int energyNew = 0;
        int energySwitched = 0;
        int i = x*width*3 + y*3;
        int iP = x_y[0]*width*3 + x_y[1]*3;

        if( !(x < 0 | y < 0 | x >= width | y>= width) ){
        
            if(y < width - 1) {
                energyNew += abs(shared_image[i] - shared_image[i + 3]) + 
                abs(shared_image[i + 1] - shared_image[i + 4]) + 
                abs(shared_image[i + 2] - shared_image[i + 5]);
                
                if(i + 3 == iP){
                    energySwitched += abs(shared_image[i] - shared_image[i + 6]) + 
                    abs(shared_image[i + 1] - shared_image[i + 7]) + 
                    abs(shared_image[i + 2] - shared_image[i + 8]);
                }
                else if(i == iP){
                    energySwitched += abs(shared_image[i] - shared_image[i + 3]) + 
                    abs(shared_image[i + 1] - shared_image[i + 4]) + 
                    abs(shared_image[i + 2] - shared_image[i + 5]);
                }
                else if(i == iP + 3){
                    energySwitched += abs(shared_image[i - 3] - shared_image[i + 3]) + 
                    abs(shared_image[i - 2] - shared_image[i + 4]) + 
                    abs(shared_image[i - 1] - shared_image[i + 5]);
                }
                else if(i + 3 == iP + 3){
                    energySwitched += abs(shared_image[i] - shared_image[i + 3]) + 
                    abs(shared_image[i + 1] - shared_image[i + 4]) + 
                    abs(shared_image[i + 2] - shared_image[i + 5]);
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
                
                if(i + 3*width == iP){
                   energySwitched +=  abs(shared_image[i] - shared_image[i + 3*width + 3]) + 
                   abs(shared_image[i + 1] - shared_image[i + 1 + 3*width + 3]) + 
                   abs(shared_image[i + 2] - shared_image[i + 2 + 3*width + 3]); 
                }
                else if(i == iP){
                    energySwitched += abs(shared_image[i+3] - shared_image[i + 3*width]) + 
                    abs(shared_image[i + 4] - shared_image[i + 1 + 3*width]) + 
                    abs(shared_image[i + 5] - shared_image[i + 2 + 3*width]);
                }
                else if(i + 3*width == iP + 3){
                    energySwitched +=  abs(shared_image[i] - shared_image[i + 3*width - 3]) + 
                    abs(shared_image[i + 1] - shared_image[i + 1 + 3*width - 3]) + 
                    abs(shared_image[i + 2] - shared_image[i + 2 + 3*width - 3]); 
                }
                else if(i == iP + 3){
                    energySwitched += abs(shared_image[i - 3] - shared_image[i + 3*width]) + 
                    abs(shared_image[i - 2] - shared_image[i + 1 + 3*width]) + 
                    abs(shared_image[i - 1] - shared_image[i + 2 + 3*width]);
                }else{
                    energySwitched += abs(shared_image[i] - shared_image[i + 3*width]) + 
                    abs(shared_image[i + 1] - shared_image[i + 1 + 3*width]) + 
                    abs(shared_image[i + 2] - shared_image[i + 2 + 3*width]);
                }
            }
           

        
        }
        
        __syncthreads();
        
        printf(" x: %d y: %d Energy: %d Energy switched: %d i: %d ip: %d |", x, y, energyNew, energySwitched, i, iP);
        
    }
    
    
""")

# __global__ for kernel functions

image = np.random.randint(0, 2, (5, 5, 3), dtype=np.uint8)
shared_image = np.zeros((5, 5, 3), dtype=np.uint8)
energy = np.array([0], dtype=np.int32)  # Use an array for energy to pass it by reference

copy_image_to_shared_memory = mod.get_function("copy_image_to_shared_memory")
image_gpu = cuda.to_device(image)
shared_image_gpu = cuda.to_device(shared_image)
energy_gpu = cuda.to_device(energy)
swap = mod.get_function("swap")

block_size = (5, 5, 1)
grid_size = (1, 1, 1)

copy_image_to_shared_memory(image_gpu, shared_image_gpu, energy_gpu, np.int32(image.shape[1]),
                            block=block_size, grid=grid_size)

print(image)
random = [1,1]
random_list = [
            [random[0], random[1]-1],  # left
            [random[0]-1, random[1]-1],  # up
            [random[0]-1, random[1]],  # up
            [random[0]-1, random[1]+1],  # up
            [random[0]-1, random[1]+2],  # up
            [random[0], random[1]+2],  # right
            [random[0]+1, random[1]-1],  # bottom
            [random[0]+1, random[1]],  # bottom
            [random[0]+1, random[1]+1],  # bottom
            [random[0]+1, random[1]+2],  # bottom
            [random[0], random[1]],
            [random[0], random[1]+1]
         ]

random_np = np.array(random_list, dtype=np.int8)
random_xy_np = np.array(random, dtype=np.int8)

random_xy_gpy = cuda.to_device(random_xy_np)
random_gpu = cuda.to_device(random_np)

print(random_np)

cuda.memcpy_dtoh(shared_image, shared_image_gpu)
cuda.memcpy_dtoh(energy, energy_gpu)
print(energy)


swap(image_gpu, random_gpu, np.int32(image.shape[1]), random_xy_gpy,
     block=(12, 1, 1), grid=(1, 1, 1))


# No need to copy energy back to the host

# print((shared_image == image).all())
#
# print("Profesor: ")
# print(energy_fun(image))
# print(image)
#
# print("Ja: ")
# print(energy[0])  # Access the value in the array
