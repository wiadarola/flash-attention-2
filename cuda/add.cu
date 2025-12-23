#include <iostream>

__global__ void add(int N, float *x, float *y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < N; i += stride)
		y[i] = x[i] + y[i];
}

int main() {
	int N = 1<<20;
	float *x, *y;

	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// int n_blocks = N / 1024 + 1;
	add<<<10, 10>>>(N, x, y);

	cudaDeviceSynchronize();

	std::cout << x[0] << " " << y[0] << std::endl;	

	cudaFree(x);
	cudaFree(y);

	return 0;
}
