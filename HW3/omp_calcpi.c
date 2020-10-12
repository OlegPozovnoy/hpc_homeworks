#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void seedThreads();
const int max_proc = 128;
const double SQUARE_MAX = (double)RAND_MAX * (double)RAND_MAX;

int main (int argc, char *argv[])
{
    const long N = 1000000000;
    long long flips=0;
    
    int tid;
    long long i;
    
    unsigned int seeds[max_proc];
    seedThreads(seeds);

#pragma omp parallel default(none) private(i,tid) shared(seeds, flips)
{
    tid = omp_get_thread_num();
    unsigned int seed = seeds[tid];
    printf("seed %d tid %d\n", seed, tid);
    
    #pragma omp for schedule(guided) reduction(+:flips)
    for (i = 0; i <= N; ++i)
    {
        double x = rand_r(&seed);
        double y = rand_r(&seed);
        if (x*x + y*y < SQUARE_MAX)
            flips++;
    }
}

printf("pi=%f after %ld iterations", flips*4./N, N);
}

void seedThreads(unsigned int* seeds){
    int tid;
    unsigned int seed;
    #pragma omp parallel private(seed, tid)
    {
        tid = omp_get_thread_num();
        seed = (unsigned) time(NULL);
        seeds[tid] = (seed & 0xFFFFFFF0) | (tid+1);
        printf("thread %d has seed %d\n", tid, seeds[tid]);
    }
}
