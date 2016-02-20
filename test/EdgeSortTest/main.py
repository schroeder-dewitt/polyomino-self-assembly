import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
from jinja2 import Environment, PackageLoader

def main():

    #Create dictionary argument for rendering
    RenderArgs= {"safe_memory_mapping":1,
                 "aligned_byte_length_genome":8,
                 "bit_length_edge_type":3, 
                 "curand_nr_threads_per_block":256,
                 "nr_tile_types":4,
                 "nr_edge_types":8,
                 "warpsize":32,
                 "fit_dim_thread_x":1,
                 "fit_dim_thread_y":1,
                 "fit_dim_block_x":1 }

    # Set environment for template package Jinja2
    env = Environment(loader=PackageLoader('main', './'))

    # Load source code from file
    Source = env.get_template('./alpha.cu') #Template( file(KernelFile).read() )
    # Render source code
    RenderedSource = Source.render( RenderArgs )

    # Save rendered source code to file
    f = open('./rendered.cu', 'w')
    f.write(RenderedSource)
    f.close()

    #Load source code into module
    KernelSourceModule = SourceModule(RenderedSource, options=None, arch="compute_20", code="sm_20")
    Kernel = KernelSourceModule.get_function("TestEdgeSortKernel")
    CurandKernel = KernelSourceModule.get_function("CurandInitKernel")

    #Initialise InteractionMatrix
    InteractionMatrix = numpy.zeros( ( 8, 8) ).astype(numpy.float32)
    def Delta(a,b):
        if a==b:
            return 1
        else:
            return 0
    for i in range(InteractionMatrix.shape[0]):
        for j in range(InteractionMatrix.shape[1]):
            InteractionMatrix[i][j] = ( 1 - i % 2 ) * Delta( i, j+1 ) + ( i % 2 ) * Delta( i, j-1 )

    #Set up our InteractionMatrix
    InteractionMatrix_h = KernelSourceModule.get_texref("t_ucInteractionMatrix")
    drv.matrix_to_texref( InteractionMatrix, InteractionMatrix_h , order="C")
    print InteractionMatrix

    #a =  numpy.random.randn(400).astype(numpy.uint8)
    #b = numpy.random.randn(400).astype(numpy.uint8)
    dest = numpy.arange(256).astype(numpy.uint8)
    for i in range(0, 256/8):
        dest[i*8 + 0] = 36
        dest[i*8 + 1] = 151
        dest[i*8 + 2] = 90
        dest[i*8 + 3] = 109
        dest[i*8 + 4] = 224
        dest[i*8 + 5] = 4
        dest[i*8 + 6] = 0
        dest[i*8 + 7] = 0
    
    dest_h = drv.mem_alloc(dest.nbytes)

    drv.memcpy_htod(dest_h, dest)
    print "before: "
    print dest
    curand = numpy.zeros(40*256).astype(numpy.uint8);
    curand_h = drv.mem_alloc(curand.nbytes)
    CurandKernel(curand_h, block=(32,1,1), grid=(1,1))
    Kernel(dest_h, curand_h, block=(32,1,1), grid=(1,1))
    drv.memcpy_dtoh(dest, dest_h)
    print "after: "
    print dest

if __name__ == '__main__':
    main()

