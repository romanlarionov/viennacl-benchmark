
Benchmarking results for Amazon C4 and G2 instance types
=======================================================

I wrote two different sets of code that tested the performance of standard BLAS level 1 and 3 operations; i.e. inner products and matrix multiplications. I ran this multiple times on local hardware (with my GTX 970) to test the validity of the code base prior to launching on EC2. I had the initial thought of using a single library to test the performace of both CPU and GPU level parallelism so I wouldn't have to write two different code sets. I found ViennaCL, which supports this functionality. 

Running the ViennaCL code lead to pretty underwhelming results. The mean results for matrix multiplication was in the range of 3 ~ 3.8 GFLOPS (With the C4 instance being higher than the G2). Some of the advertised GFLOPs rates, found online, mark standard BLAS operations in the hundereds, or THOUSANDS, of GLFOPS. Looking at the original ViennaCL [paper](http://www.iue.tuwien.ac.at/pdf/ib_2010/Rupp_GPUScA.pdf), I found their publicized GFLOPS for sparse matrix multiplications being around 1.1 on a GTX 470. They list comperable performace for their CPU abstraction. This raised a red flag, as I find that those values should be much higher.

I decided to test CUDA's BLAS library, cuBLAS, as well. The mean GLFOPS rate for this being ~89 GFLOPS. While this is substantially higher than the ViennaCL implementation, it still falls short of the 0.6 TFLOPS I thought was possible. 

It just goes to show that these performance tests are very much application specific. My assumption is that ViennaCL is fairly optimized for sparse matrix operations, but incurs overhead when dealing with dense matrices because it cant apply a compression algorithm. I haven't tested a purely pthread or OpenMP solution, but I have a feeling like it will be significantly worst and the cuBLAS implementation.

As far as why these reported GFLOPS rates are so low, I have come to the conclusion that the benchmark reports seen online are for theoretical performance only. The way to calculate the theoretical performance of an nVidia GPU for double precision operations is denote by this equation:

```
<num cuda cores> x <clock rate> x <2 double precision multiply-add operations> = <GFLOPS rate>
```

For the G2 instance GPUs, the theoretical GLFOPS rate per GPU is:

```
<1536 cores> x <800 MHz> x <2> = 2457.6 GFLOPS
```

which is pretty close to the reported rate that I found [online](http://www.techpowerup.com/gpudb/2312/grid-k520).

In conclusion, GPU and CPU real time calculation speeds on C4 and G2 instance types are going to be much slower than I initially expected. Outside of bus data transfer times, GPUs also incur a price in poor instruction allocation on the architecture level. This is not something I was expecting to be such a big deal. 

I still want to cook up some alternate pthreads or OpenMP test to rival cuBLAS, but I feel as though I've personally spent too long on this already. If anyone would want to help with this, I'd greatly appreciate it. 
