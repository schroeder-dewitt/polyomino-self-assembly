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
                 "nr_tile_types":8,
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
    Kernel = KernelSourceModule.get_function("TestMovelistKernel")

    #a =  numpy.random.randn(400).astype(numpy.uint8)
    #b = numpy.random.randn(400).astype(numpy.uint8)
    dest = numpy.arange(256).astype(numpy.uint8)
    dest[0] = 36
    dest[1] = 151
    dest[2] = 90
    dest[3] = 109
    dest[4] = 224
    dest[5] = 4
    dest[6] = 0
    dest[7] = 0
    dest_h = drv.mem_alloc(dest.nbytes)
    drv.memcpy_htod(dest_h, dest)
    print "before: "
    print dest
    Kernel(dest_h, block=(32,1,1), grid=(1,1))
    drv.memcpy_dtoh(dest, dest_h)
    print "after: "
    print dest

if __name__ == '__main__':
    main()

