//
// Created by oleg on 15.10.2020.
//

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include "matrix_utils.h"
#include "main_pagerank.h"


// Based on
// https://yadi.sk/i/9WjGG7YCswvzA
int main() {
    double *A = get_article_matrix();
    int N = 6;

    /*double *A = get_koshey_matrix();
    int N = 8;*/

    double d = 1.;
    //double d = .85;
    printf("\n\nInitial matrix\n");
    print_matrix(A, N);

    normalize_matrix(A, N, d);
    printf("\n\nNormalized matrix, d=%f\n", d);
    print_matrix(A, N);

    double *result = get_eugene(A, N, 0.01);
    free(A);
    printf("\n\nThe final PageRank\n");
    print_vector(result, N);
    free(result);
}


double *matrix_vector_mult(const double *A, const double *b, int N) {
    double *result = (double *) malloc(sizeof(double) * N);
    int i;

#pragma omp parallel default(none) private(i) shared(A, b, N, result)
    {
#pragma omp parallel for
        for (i = 0; i < N; i++) {
            result[i] = 0;
            for (int j = 0; j < N; j++) {
                result[i] += A[i * N + j] * b[j];
            }
        }
    }
    return result;
}


void normalize_matrix(double *A, int N, double d) {
    double sum[N];
    int i, tid;

#pragma omp parallel default(none) private(i, tid) shared(A, N, sum, d)
    {
        tid = omp_get_thread_num();
#pragma omp for
        for (i = 0; i < N; i++) {
            //printf("\nstep1: thread %d i=%d", tid, i);
            sum[i] = 0;

            for (int j = 0; j < N; j++) {
                //printf("\nstep2: thread %d i=%d j=%d", tid, i, j);
                sum[i] += A[j * N + i];
            }

            for (int j = 0; j < N; j++) {
                //printf("\nstep3: thread %d i=%d j=%d", tid, i, j);
                if (sum[i] != 0)
                    A[j * N + i] /= sum[i] / d;
                else
                    A[j * N + i] = d / N;

                A[j * N + i] += (1 - d) / N;
            }
        }
    }
}


void normalize_vector(double *v, int N) {
    double norm = 0;
    print_vector(v, N);
    int i;
#pragma omp parallel default(none) private(i) shared(v, N, norm)
    {
#pragma omp for reduction(+:norm)
        for (i = 0; i < N; i++)
            norm += fabs(v[i]);
    }

    printf("\nnorm %f\n", norm);

#pragma omp parallel default(none) private(i) shared(v, N, norm)
    {
#pragma omp for
        for (i = 0; i < N; i++)
            v[i] /= norm;
    }
}


double get_vector_diff_norm(double *v1, double *v2, int N) {
    double norm = 0;
    int i;
#pragma omp parallel default(none) private(i) shared(v1, v2, N, norm)
    {
#pragma omp for reduction(+:norm)
        for (i = 0; i < N; i++)
            norm += fabs(v1[i] - v2[i]);
    }
    return norm;
}


double *get_eugene(double *A, int N, double thr) {
    double *result = (double *) malloc(sizeof(double) * N);
    double *result_tmp;
    double error = 1;
    int i;

#pragma omp parallel default(none) private(i) shared(N, result)
    {
#pragma omp parallel for
        for (i = 0; i < N; i++)
            result[i] = 1. / N;
    }

    printf("\ninitial vector\n");
    print_vector(result, N);
    int step = 1;
    while (error > thr) {
        printf("\n step %d", step);
        normalize_vector(result, N);
        result_tmp = matrix_vector_mult(A, result, N);
        printf("\n after multiplication\n");
        print_vector(result_tmp, N);
        normalize_vector(result_tmp, N);    // при правильных матрицах избыточно, но гарантий нет, так что пусть будет
        printf("\n after normaliztion\n");
        print_vector(result_tmp, N);
        error = get_vector_diff_norm(result_tmp, result, N);
        printf("\n error %f", error);
        memcpy(result, result_tmp, N * sizeof(double));
        free(result_tmp);
        step++;
        if (step > 1000) {
            printf("\nThe procedure hasn't finished in 1000 steps and will be terminated.\n");
            break;
        }
    }
    return result;
}

double *get_koshey_matrix() {
    int N = 8;
    double *A = (double *) calloc(N * N, sizeof(double));
    A[0 * N + 4] = 1;
    A[1 * N + 0] = 1;
    A[2 * N + 0] = 1;
    A[2 * N + 1] = 1;
    A[3 * N + 0] = 1;
    A[3 * N + 5] = 1;
    A[5 * N + 2] = 1;
    A[6 * N + 5] = 1;
    A[7 * N + 0] = 2;
    A[7 * N + 6] = 1;
    return A;
}


// Based on
// https://yadi.sk/i/9WjGG7YCswvzA
double *get_article_matrix() {

    int N = 6;
    double *A = (double *) calloc(N * N, sizeof(double));

    A[0 * N + 3] = 1;
    A[0 * N + 4] = 1;
    A[0 * N + 5] = 1;

    A[1 * N + 0] = 1;
    A[1 * N + 5] = 1;

    A[2 * N + 0] = 1;
    A[2 * N + 1] = 1;
    A[2 * N + 3] = 1;

    A[3 * N + 1] = 1;
    A[3 * N + 2] = 1;

    A[4 * N + 1] = 1;
    A[4 * N + 2] = 1;
    A[4 * N + 3] = 1;
    A[4 * N + 5] = 1;

    A[5 * N + 0] = 1;
    A[5 * N + 1] = 1;
    A[5 * N + 3] = 1;

    return A;
}
