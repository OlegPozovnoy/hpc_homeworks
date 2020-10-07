//
// Created by oleg on 06.10.2020.
//

#ifndef MATRIXMULT_MAIN_H
#define MATRIXMULT_MAIN_H

#include<vector>

using std::vector;

vector<vector<double>> matrix_mult(vector<vector<double>> A, vector<vector<double>> B);
vector<double> vect_matrix_mult(vector<double> a, vector<vector<double>> B);

vector<vector<double>> create_matrix(const int& x_dim, const int& y_dim);
vector<double> create_vector(const int& dim);

#endif //MATRIXMULT_MAIN_H
