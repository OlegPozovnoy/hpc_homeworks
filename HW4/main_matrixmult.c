//
// Created by oleg on 20.10.2020.
//
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "main_matrixmult.h"
#include "matrix_utils.h"


int main() {
    int N = 16;
    int power = 5;
    double *A = getMatrixPowerTest(N);
    print_matrix(A, N);
    double *C = matrix_power(A, N, power);
    printf("\nFinal result:\n");
    print_matrix(C, N);
    free(A);
    free(C);
}


double *matrix_power(double *A, int N, int power) {
    int start = 31;
    while (start > 0 && ((1 << start) & power) == 0)
        start--;

    //printf("\n start = %d %d \n", start, power);

    double *tmp = (double *) malloc(N * N * sizeof(double));
    memcpy(tmp, A, N * N * sizeof(double));
    double *nxt;
    int current_power = 1;

    for (int i = start - 1; i >= 0; i--) {
        printf("\nSquaring the matrix\n");
        current_power *= 2;
        nxt = square_matrix_mult(tmp, tmp, N);
        printf("\nCurrent power %d.\n", current_power);
        print_matrix(nxt, N);
        free(tmp);
        tmp = nxt;
        if (((1 << i) & power) > 0) {
            printf("\nMult result by initial matrix\n");
            nxt = square_matrix_mult(tmp, A, N);
            current_power++;
            printf("\nCurrent power %d.\n", current_power);
            print_matrix(nxt, N);
            free(tmp);
            tmp = nxt;
        }
    }
    return tmp;
}


double *getMatrixPowerTest(int N) {
    double *A = (double *) calloc(N * N, sizeof(double));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            if (abs(i - j) % N == 1 || abs(i - j) % N == N - 1)
                A[i * N + j] = 1;
    }
    return A;
}


double *square_matrix_mult(const double *A, const double *B, int N) {
    double *result = (double *) malloc(sizeof(double) * N * N);
    int i, j, tid;
#pragma omp parallel private(i, j, tid) shared(A, B, N, result)
    {
        //tid = omp_get_thread_num();
#pragma omp for collapse(2)  schedule(guided)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                //printf("\nthread=%d (i,j) %d %d", tid, i, j);
                result[i * N + j] = 0;
                for (int k = 0; k < N; k++) {
                    result[i * N + j] += A[i * N + k] * B[k * N + j];
                }
            }
        }
    }
    return result;
}