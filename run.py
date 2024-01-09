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
""")

# __global__ for kernel functions

image = np.random.randint(0, 2, (2, 2, 3), dtype=np.uint8)
shared_image = np.zeros((2, 2, 3), dtype=np.uint8)
energy = np.array([0], dtype=np.int32)  # Use an array for energy to pass it by reference

copy_image_to_shared_memory = mod.get_function("copy_image_to_shared_memory")
image_gpu = cuda.to_device(image)
shared_image_gpu = cuda.to_device(shared_image)
energy_gpu = cuda.to_device(energy)

block_size = (2, 2, 1)
grid_size = (1, 1, 1)

copy_image_to_shared_memory(image_gpu, shared_image_gpu, energy_gpu, np.int32(image.shape[1]),
                            block=block_size, grid=grid_size)

cuda.memcpy_dtoh(shared_image, shared_image_gpu)
cuda.memcpy_dtoh(energy, energy_gpu)
# No need to copy energy back to the host

print((shared_image == image).all())

print("Profesor: ")
print(energy_fun(image))
print(image)

print("Ja: ")
print(energy[0])  # Access the value in the array
