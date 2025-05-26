// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define Nx 256
#define Ny 256
#define h 1.0
#define nstep 10000
#define nprint 100
#define dt 0.01
#define c0 0.4
#define mob 1.0
#define grad_coef 0.5

double t = 0;
double c[Nx][Ny];
double f[Nx][Ny];

void init()
{
    //initiate c[][] with noise
    srand(time(0));
    double noise = 0.02;
    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++)
            c[i][j] = c0 + ((rand() % 10001 - 5000) / 10000.0) * noise;
    return;
}
double free_energy(double c)
{
    //calculate free energy
    double result;
    double A = 1.0;
    result = A * (2.0 * c * pow(1 - c, 2) - 2 * pow(c, 2) * (1 - c)); //杩欓噷鍙兘鏄姞鍙凤紝鍙兘鍙互鍖栫畝
    return result;
}
void print(FILE* file)
{
    //save data from c to file
    fprintf(file,
        "VARIABLES=\"x\",\"y\",\"c\"\n"
        "ZONE I=%d,J=%d,F=POINT\n",
        Nx, Ny);
    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++)
            fprintf(file, "%d    %d    %lf\n", i, j, c[i][j]);
}
double claplace(int x, int y)
{
    //laplace for c
    double lap;
    int r, l, u, d;
    r = (x + 1) % Nx;
    l = (x - 1 + Nx) % Nx;
    u = (y - 1 + Ny) % Ny;
    d = (y + 1) % Ny;
    double n, e, s, w, ne, nw, se, sw, center;
    center = c[x][y];
    n = c[x][u];
    e = c[r][y];
    s = c[x][d];
    w = c[l][y];
    ne = c[r][u];
    nw = c[l][u];
    se = c[r][d];
    sw = c[l][d];
    lap = (((n + e + s + w) * 4 + (ne + nw + se + sw)) - 20 * center) / (6 * h * h);
    return lap;
}
double flaplace(int x, int y)
{
    //laplace for f
    double lap;
    int r, l, u, d;
    r = (x + 1) % Nx;
    l = (x - 1 + Nx) % Nx;
    u = (y - 1 + Ny) % Ny;
    d = (y + 1) % Ny;
    double n, e, s, w, ne, nw, se, sw, center;
    center = f[x][y];
    n = f[x][u];
    e = f[r][y];
    s = f[x][d];
    w = f[l][y];
    ne = f[r][u];
    nw = f[l][u];
    se = f[r][d];
    sw = f[l][d];
    lap = (((n + e + s + w) * 4 + (ne + nw + se + sw)) - 20 * center) / (6 * h * h);
    return lap;
}

int main()
{
    clock_t start = clock();
    init();
    char filename[100];
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads / 2);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Hello from thread %d\n", thread_id);

#pragma omp for
        for (int i = 10000; i < 10000 + nstep; i++)
        {
            for (int x = 0; x < Nx; x++)
                for (int y = 0; y < Ny; y++)
                    f[x][y] = free_energy(c[x][y]) - grad_coef * claplace(x, y);
            for (int x = 0; x < Nx; x++)
                for (int y = 0; y < Ny; y++)
                {
                    c[x][y] = c[x][y] + dt * mob * flaplace(x, y);
                    if (c[x][y] > 0.9999)
                        c[x][y] = 0.9999;
                    if (c[x][y] < 0.0001)
                        c[x][y] = 0.0001;
                }

            if (i % nprint == 0)
            {
                snprintf(filename, 100, "Data/time_%d.dat", i);
                FILE* file;
                fopen_s(&file, filename, "w");
                if (file == NULL) {
                    printf("File %s Not Found.\n", filename);
                    continue; //
                }
                print(file);
                fclose(file);
            }
        }
    }
    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("程序运行时间: %.6f 秒\n", duration);

    return 0;
}

/*
example of Time_XXXXX.dat
    VARIABLES="x","y","c"
    ZONE I=64,J=64,F=POINT
    0    0    0.037109
    0    1    0.040857
    0    2    0.082509
*/


/*
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    int myid, numprocs, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Get_processor_name(processor_name, &namelen);
    if (myid == 0)
    {
        printf("number of processes: %d\n", numprocs);
    }
    int des = (myid + 1) % 8;
    int rec = (myid + 7) % 8;
    int mailbox;
    MPI_Send(&myid, 1, MPI_INT, des, 1, MPI_COMM_WORLD);
    MPI_Recv(&mailbox, 1, MPI_INT, rec, 1, MPI_COMM_WORLD, &status);
    printf("%s: Hello world from process %d, I received a message from process %d \n", processor_name, myid, mailbox);

    MPI_Finalize();

    return 0;

}
//mpiexec -n 8 MSMPI.exe
*/