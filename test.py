import pyopencl as cl
import pyopencl.array
import pyopencl.tools
import numpy as np

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

# Standard numpy struct
my_struct = np.dtype([("field1", np.int32), ("field2", np.float32)])

# Map the numpy struct to a C struct
my_struct, my_struct_c_decl = cl.tools.match_dtype_to_c_struct(ctx.devices[0], "my_struct", my_struct)

# Prints the syntax for defining our cstruct
# print(my_struct_c_decl)

# Inform pyopencl about our struct
my_struct = cl.tools.get_or_register_dtype("my_struct", my_struct)

# Load up a numpy array of our structs with some data
ary_host = np.empty(20, my_struct)
ary_host["field1"].fill(217)
ary_host["field2"].fill(1000)
ary_host[13]["field2"] = 12
print("Numpy Array:")
print(ary_host)
print() 

# Transfer array to device
ary = cl.array.to_device(queue, ary_host)

# Define a kernel
prg = cl.Program(ctx, my_struct_c_decl + """
  __kernel void set_to_1(__global my_struct *a) {
      a[get_global_id(0)].field1 = 1;
  }""").build()

# Not sure what None here is for
# Running our own Kernel on the device
evt = prg.set_to_1(queue, ary.shape, None, ary.data)

print("Device array after calling function: ")
print(ary) 
print()

from pyopencl.elementwise import ElementwiseKernel

# Using pyopencl's element wise
elwise = ElementwiseKernel(ctx, "my_struct *a", "a[i].field1 = -2; a[i].field2 = -1;", preamble=my_struct_c_decl)
evt = elwise(ary)
print("Device array after using elementwise: ")
print(ary)
print()