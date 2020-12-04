#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void task1(const char* filename_in, const char* filename_out, double* filter);
void task2(const char* filename_in, const char* filename_out, int window_size_h, int window_size_w);
void task3(const char* filename_in, const char* filename_out);

const int filter_size = 5;


__global__ void task1CUDA(uchar* src, int* d_params, double* filter) {
	int rows = d_params[0];
	int cols = d_params[1];
	int channels = d_params[2];

	int index = blockDim.x * blockIdx.x + threadIdx.x;

	double s = 0;
	for (int k = 0; k < filter_size * filter_size; k++)
		s += filter[k];

	int ch = index % channels;
	index = (index - ch) / channels;
	int row = (index) % rows;
	int col = (index) / rows;

	if (col < cols) {
			double sum = 0;
			for (int k = -2; k <= 2; k++)
				for (int l = -2; l <= 2; l++) {
					int srcRow = (row + k) < 0 ? 0 : row + k;
					srcRow = srcRow >= rows ? rows - 1 : srcRow;

					int srcCol = (col + l) < 0 ? 0 : col + l;
					srcCol = (srcCol >= cols) ? cols - 1 : srcCol;

					sum += (double)(filter[k + 2 + 5 * (l + 2)] * src[(srcRow * cols + srcCol) * channels + ch]) / s;
				}
			__syncthreads();
			src[(row * cols + col) * channels + ch] = (int)sum;
	}
}


__global__ void task2CUDA(uchar* src, int* d_params) {
	int rows = d_params[0];
	int cols = d_params[1];
	int channels = d_params[2];

	int window_h = d_params[3];
	int window_w = d_params[4];

	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int ch = index % channels;
	int row = ((index - ch) / channels) % rows;
	int col = ((index - ch) / channels) / rows;

	if (col < cols) {
		//double sum = 0;
		size_t size = (2 * window_h + 1) * (2 * window_w + 1);
		uchar* values = (uchar*)malloc(size * sizeof(uchar));
		int count = 0;
		for (int k = -window_w; k <= window_w; k++)
			for (int l = -window_h; l <= window_h; l++) {
				int srcRow = (row + k) < 0 ? 0 : row + k;
				srcRow = srcRow >= rows ? rows - 1 : srcRow;

				int srcCol = (col + l) < 0 ? 0 : col + l;
				srcCol = (srcCol >= cols) ? cols - 1 : srcCol;

				values[count] = (src[(srcRow * cols + srcCol) * channels + ch]);
				count++;
			}

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = j+1; k < size; k++) {
					if (values[k] < values[j]) {
						uchar tmp = values[k];
						values[k] = values[j];
						values[j] = tmp;
					}
				}
			}
		}
		__syncthreads();
		src[(row * cols + col) * channels + ch] = values[size/2];
		free(values);
	}
}


__global__ void task3DumbVeresion(int* histohram, uchar* src, int* d_params) {
	int rows = d_params[0];
	int cols = d_params[1];
	int channels = d_params[2];

	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int ch = index % channels;
	int row = ((index - ch) / channels) % rows;
	int col = ((index - ch) / channels) / rows;

	if (col < cols) {
		uchar val = (src[(row * cols + col) * channels + ch]);
		atomicAdd(&(histohram[256 * ch + val]), 1);
	}
}


int main(int argc, char** argv)
{
	const char* filename_in = "lions.jpg";
	const char* filename_out1_1 = "lions_out1_1.jpg";
	const char* filename_out1_2 = "lions_out1_2.jpg";
	const char* filename_out2 = "lions_median_filter.jpg";
	const char* filename_out3 = "histohram.txt";

	double h_filter1[filter_size * filter_size] =
	{1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1};

	double h_filter2[filter_size * filter_size] =
	{ 1, 1, 1, 1, 1,
	1, 0, 0, 0, 1,
	1, 0, 0, 0, 1,
	1, 0, 0, 0, 1,
	1, 1, 1, 1, 1 };

	task1(filename_in, filename_out1_1, h_filter1);
	task1(filename_in, filename_out1_2, h_filter2);

	task2(filename_in, filename_out2, 2, 2);

	task3(filename_in, filename_out3);

	return 0;
}


void task1(const char* filename_in, const char* filename_out, double* h_filter) {

	Mat src = imread(samples::findFile(filename_in), IMREAD_COLOR);

	double* d_filter1;
	uchar* d_source;

	int h_params[3] = { src.rows , src.cols, src.channels() };
	int* d_params;
	cudaMalloc(&d_params, 3 * sizeof(int));
	cudaMemcpy(d_params, h_params, 3 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_source, src.rows * src.cols * src.channels() * sizeof(uchar));
	cudaMalloc(&d_filter1, filter_size * filter_size * sizeof(double));

	cudaMemcpy(d_filter1, h_filter, filter_size * filter_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_source, src.data, src.rows * src.cols * src.channels() * sizeof(uchar), cudaMemcpyHostToDevice);

	printf("\nStartCUDA");
	int blocks = src.rows * src.cols * src.channels() / 1024 + 1;
	task1CUDA << <blocks, 1024 >> > (d_source, d_params, d_filter1);
	printf("\nEndCUDA");

	uchar* destbuffer = (uchar*)malloc(src.rows * src.cols * src.channels() * sizeof(uchar));

	cudaMemcpy(destbuffer, d_source, src.rows * src.cols * src.channels() * sizeof(uchar), cudaMemcpyDeviceToHost);
	for (int i = 0; i < src.rows * src.cols * src.channels(); i++) {
		src.data[i] = destbuffer[i];
	}

	imwrite(filename_out, src);

	cudaDeviceSynchronize();
	cudaFree(d_filter1);
	cudaFree(d_source);
	cudaFree(d_params);
	free(destbuffer);
}


void task2(const char* filename_in, const char* filename_out, int window_size_h, int window_size_w) {

	Mat src = imread(samples::findFile(filename_in), IMREAD_COLOR);
	uchar* d_source;

	int h_params[5] = { src.rows , src.cols, src.channels(), window_size_h, window_size_w };
	int* d_params;
	cudaMalloc(&d_params, 5 * sizeof(int));
	cudaMemcpy(d_params, h_params, 5 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_source, src.rows * src.cols * src.channels() * sizeof(uchar));
	cudaMemcpy(d_source, src.data, src.rows * src.cols * src.channels() * sizeof(uchar), cudaMemcpyHostToDevice);

	printf("\nStartCUDA");
	int blocks = src.rows * src.cols * src.channels() / 1024 + 1;
	task2CUDA << <blocks, 1024 >> > (d_source, d_params);
	printf("\nEndCUDA");

	uchar* destbuffer = (uchar*)malloc(src.rows * src.cols * src.channels() * sizeof(uchar));
	printf("\ncreateBuffer");
	cudaMemcpy(destbuffer, d_source, src.rows * src.cols * src.channels() * sizeof(uchar), cudaMemcpyDeviceToHost);
	for (int i = 0; i < src.rows * src.cols * src.channels(); i++) {
		src.data[i] = destbuffer[i];
	}
	printf("\ncopyResult");
	imwrite(filename_out, src);

	cudaDeviceSynchronize();
	cudaFree(d_source);
	cudaFree(d_params);
	free(destbuffer);
}


void task3(const char* filename_in, const char* filename_out) {

	Mat src = imread(samples::findFile(filename_in), IMREAD_COLOR);
	uchar* d_source;

	int h_params[3] = { src.rows , src.cols, src.channels()};
	int* d_params;
	cudaMalloc(&d_params, 3 * sizeof(int));
	cudaMemcpy(d_params, h_params, 3 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_source, src.rows * src.cols * src.channels() * sizeof(uchar));
	cudaMemcpy(d_source, src.data, src.rows * src.cols * src.channels() * sizeof(uchar), cudaMemcpyHostToDevice);

	int* h_histohram = (int*)calloc( src.channels()*256, sizeof(int));
	int* d_histohram;

	cudaMalloc(&d_histohram, 256 * src.channels() * sizeof(int));
	cudaMemcpy(d_histohram, h_histohram, 256 * src.channels() * sizeof(int), cudaMemcpyHostToDevice);

	printf("\nStartCUDA");
	int blocks = src.rows * src.cols * src.channels() / 1024 + 1;
	task3DumbVeresion << <blocks, 1024 >> > (d_histohram, d_source, d_params);
	printf("\nEndCUDA");

	cudaMemcpy(h_histohram, d_histohram, 256 * src.channels() * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\ncopyResult");

	int sum = 0;
	FILE* f = fopen(filename_out, "w");
	for (int c = 0; c < src.channels(); c++)
		for (int i = 0; i < 256; i++) {
			sum += h_histohram[c * 256 + i];
			fprintf(f, "\nchannel: %d intensity: %d count: %d", c, i, h_histohram[c * 256 + i]);
		}
	fclose(f);

	printf("compare: %d %d", sum, src.rows * src.cols * src.channels());

	cudaDeviceSynchronize();
	cudaFree(d_source);
	cudaFree(d_params);
	cudaFree(d_histohram);
	free(h_histohram);
}