//
// Created by oleg on 15.10.2020.
//

#ifndef HW4_MAIN_PAGERANK_H
#define HW4_MAIN_PAGERANK_H

double *matrix_vector_mult(const double *A, const double *b, int N);
void normalize_matrix(double *A, int N, double d);
void normalize_vector(double *v, int N);
double get_vector_diff_norm(double *v1, double *v2, int N);
double *get_eugene(double *A, int N, double thr);
double *get_koshey_matrix();
double *get_article_matrix();

#endif //HW4_MAIN_PAGERANK_H
