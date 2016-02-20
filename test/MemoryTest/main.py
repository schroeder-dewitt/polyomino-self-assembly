import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
struct B{
    struct {
       float val[100];
    } data;
    __device__ void set (float val){
        for(int i=0;i<100;i++){
            this->data.val[i] = val;
	}
    }
};

struct A{
    struct {
      B array[10];
    } data;
    __device__ void access(float val){
        for(int i=0;i<10;i++){
            this->data.array[i].set(val);                  
        }
    } 
};

__global__ void multiply_them(float *dest, float *a, float *b)
{
   __shared__ A test;
   test.access(99.0f);
   dest[0] = test.data.array[5].data.val[0];
   dest[1] = test.data.array[5].data.val[1];
   dest[3] = 6.0f;
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))

print dest
