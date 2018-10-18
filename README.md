Pyopencl Stuffs

# Andrew Lecture Notes
- OpenCL is really an extension of C++
- Abstractions (PLatform, device queue, kernel, buffer)
  - Platform: Collection of devices (CPU: d0, d1 ..., GPU: d0, d1 - eg 2 physical graphics cards)
    - Each platform contains collection of devices (eg a platform for each of the GPUs in a machine and a platform for each of the cpus in the machine)
    - Devices grouped inside of a platform have similar compute capabilities
    - Really big machienes: may have multiple platforms
    - CLuster system may distribute across network
- Each device has its own command queue (bound to a specific device)
  - Execute serially by default
  - Parallel: set queue in out of order mode
  - Submit kernels to queue
- Sync: eg results of A and B required to compute C
  - Push A to Q0, B to Q1 --> q0.wait(); q1.wait(); --> submit C to Q2
  - Can also wait on specifc outputs from A / B etc
- Buffers: represents data in different places 
  - Can wire up one of task C's inputs to be B's output buffer
  - Specify _where_ the buffer should exist (as oppsed to maloc which just uses "memory")
- MESA is a decent standard opencl driver for a bunch of platforms  

- For each integral point in the complex plane (within the resolution boundary)
  - Check if its in mand set --> embarisingly parallel
  - Split it into chunks
    - But break it into *contiguous* chunks (sroted in memory)
      - So do it a row at a time
      - No data dependencies between points, so work division is actually arbitrary
      - But work per chunk depends on data (might need 1 --> max iterations)
      - Might want to do some scheduling
         - For chunks that aren't converging move them to GPU and run a bunch of iterations

- Nvidia processors:
  - Each core (eg 2500) (Shader) can run 32 threads
  - But each must be running same instructions (same function)
  - Thats why we define kernels in terms of a **single** unit of work 

- For getting benefit from chunks (in a case where we have multiple device)
  - Just make multiple queues (on for each of the devices in the platform)

- Benefit of OpenCL:
  - Write kernel once, run anywhere that has an opencl driver implmentation