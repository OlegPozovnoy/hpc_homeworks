//
// Created by oleg on 20.10.2020.
//

#include "matrix_utils.h"


void print_matrix(const double *A, int N) {
    for (int i = 0; i < N; i++) {
        printf("\n");
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
    }
    printf("\n");
}


void print_vector(const double *A, int N) {
    printf("\n");
    for (int j = 0; j < N; j++) {
        printf("%.2f ", A[j]);
    }
    printf("\n");
}