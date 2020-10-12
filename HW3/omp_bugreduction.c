#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


float dotprod(float * a, float * b, size_t N)
{
    int tid;
    float sum;
    int i;
    #pragma omp parallel default(none) private(tid,i) shared(sum, a, b, N)
    {
    tid = omp_get_thread_num();
    #pragma omp for reduction(+:sum)
    for (i = 0; i < N; ++i)
        {
        sum += a[i] * b[i];
        printf("tid = %d i = %d\n", tid, i);
        }
    }
    return sum;
}


int main (int argc, char *argv[])
{
    const size_t N = 100;
    int i;
    float sum;
    float a[N], b[N];

    #pragma omp parallel for default(none) private(i) shared(a,b)
    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = (double)i;
    }

    sum = dotprod(a, b,  N);
    printf("Sum = %f\n",sum);

    return 0;
}

