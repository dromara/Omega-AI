extern "C"
__global__ void add(int n, float *a, float *b, float *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
    	for(int j = 0;j<5000;j++){
    		sum[i] = a[i] + b[i];
    	}
    }

}
