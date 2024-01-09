import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""

    __global__ void copy_image_to_shared_memory(unsigned char *image, unsigned char *shared_image, int width, 
    int height ) { 
        int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
        int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (tid_x < width && tid_y < height) {
          int index = tid_y * width * 3 + tid_x * 3;
          int shared_index = threadIdx.y * blockDim.x * 3 + threadIdx.x * 3;
    
          shared_image[shared_index] = image[index];
          shared_image[shared_index + 1] = image[index + 1];
          shared_image[shared_index + 2] = image[index + 2];
        }
    }

""")
# __global__ for kernel functions

image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
shared_image = np.zeros((32, 32, 3), dtype=np.uint8)
print(image)
print("asdasdsad")

copy_image_to_shared_memory = mod.get_function("copy_image_to_shared_memory")
image_gpu = cuda.to_device(image)
shared_image_gpu = cuda.to_device(shared_image)

block_size = (16, 16, 1)
grid_size = ((image.shape[1] + block_size[0] - 1) // block_size[0], (image.shape[0] + block_size[1] - 1) // block_size[1], 1)

copy_image_to_shared_memory(image_gpu, shared_image_gpu, np.int32(image.shape[1]), np.int32(image.shape[0]),
                            block=block_size, grid=grid_size)

cuda.memcpy_dtoh(shared_image, shared_image_gpu)

print(shared_image)

