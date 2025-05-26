#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

#define Nx 128
#define Ny 128
#define h 1.0
#define nstep 50000
#define nprint 1000
#define dt 0.01
#define c0 0.4
#define mob 1.0
#define grad_coef 0.4
#define noise 0.02

void init(double c[][Ny])
{
    //初始化
    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++)
            c[i][j] = c0 + ((rand() % 10001 - 5000) / 10000.0) * noise;
    return;
}

__device__ double free_energy(double c)
{
    //计算自由能
    double result;
    double A = 1.0;
    result = A * (2.0 * c * pow(1 - c, 2) - 2 * pow(c, 2) * (1 - c));
    return result;
}

void print(double c[][Ny], int step)
{
    char filename[100] = {};
    sprintf(filename, "Data\\time_%d.dat", step);
    FILE* file = fopen(filename, "w");
    fprintf(file,
        "VARIABLES=\"x\",\"y\",\"c\"\n"
        "ZONE I=%d,J=%d,F=POINT\n",
        Nx, Ny);

    for (int i = 0; i < Nx; i++)   //输入当前进程分块的序参量的信息
        for (int j = 0; j < Ny; j++)
            fprintf(file, "%d    %d    %lf\n", i, j, c[i][j]);
    fclose(file);
    return;
}

__device__ double laplace(int x, int y, double a[][Ny])
{
    //计算矩阵a在(x,y)处的拉普拉斯算子
    double lap;
    int r, l, u, d;
    r = (x + 1) % Nx;
    l = (x - 1 + Nx) % Nx;
    u = (y - 1 + Ny) % Ny;
    d = (y + 1) % Ny;
    double n, e, s, w, ne, nw, se, sw, center;
    center = a[x][y];
    n = a[x][u];
    e = a[r][y];
    s = a[x][d];
    w = a[l][y];
    ne = a[r][u];
    nw = a[l][u];
    se = a[r][d];
    sw = a[l][d];
    lap = (((n + e + s + w) * 4 + (ne + nw + se + sw)) - 20 * center) / (6 * h * h);
    return lap;
}

__global__ void cCal(double c[][Ny], double f[][Ny])
{
    int tix = threadIdx.x;
    int bix = blockIdx.x;
    f[bix][tix] = free_energy(c[bix][tix]) - grad_coef * laplace(bix, tix, c);
    return;
}

__global__ void fCal(double c[][Ny], double f[][Ny])
{
    int tix = threadIdx.x;
    int bix = blockIdx.x;
    c[bix][tix] = c[bix][tix] + dt * mob * laplace(bix, tix, f);
    return;
}

int main(int argc, char* argv[])
{
    double c[Nx][Ny];
    clock_t start = clock();
    
    init(c);
    double (*d_c)[Ny], (*d_f)[Ny];
    cudaMalloc(&d_c, sizeof(double) * Nx * Ny);
    cudaMalloc(&d_f, sizeof(double) * Nx * Ny);
    cudaMemcpy(d_c, c, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);
    dim3 grid(Nx, 1, 1), block(Ny, 1, 1);

    for (int step = 0; step <= nstep; step++)
    {
        fCal<<<grid, block>>>(d_c, d_f);
        cCal<<<grid, block>>>(d_c, d_f);
        cudaMemcpy(c, d_c, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost);
        if(step%nprint==0)
        {
            cudaMemcpy(c, d_c, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost);
            print(c, step);
        }
    }

    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total time: %.2fs\n", duration);
    cudaFree(d_c);
    cudaFree(d_f);
    return 0;
}

