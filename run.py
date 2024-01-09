import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    __global__ void copy_image_to_shared_memory(unsigned char *image, unsigned char *shared_image, int width, int height) {
        int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
        int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    
        if (idx_x < width && idx_y < height) {
            int idx_image = (idx_y * width + idx_x) * 3; // 3 channels
            int idx_shared = (threadIdx.y * blockDim.x + threadIdx.x) * 3; // 3 channels
    
            shared_image[idx_shared] = image[idx_image];
            shared_image[idx_shared + 1] = image[idx_image + 1];
            shared_image[idx_shared + 2] = image[idx_image + 2];
        }
    }
""")

image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
shared_image = np.zeros((32, 32, 3), dtype=np.uint8)

copy_image_to_shared_memory = mod.get_function("copy_image_to_shared_memory")
image_gpu = cuda.to_device(image)
shared_image_gpu = cuda.to_device(shared_image)

block_size = (32, 32, 1)
grid_size = (1, 1, 1)

print(image)

copy_image_to_shared_memory(image_gpu, shared_image_gpu, np.int32(image.shape[1]), np.int32(image.shape[0]),
                            block=block_size, grid=grid_size)

cuda.memcpy_dtoh(shared_image, shared_image_gpu)
print("\n\n\n\n")
print(shared_image)
