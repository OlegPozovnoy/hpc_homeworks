//
// Created by oleg on 06.10.2020.
//

#include "main.h"
#include <cstdlib>
#include <cassert>

vector<vector<double>> create_matrix(const int& x_dim,const int& y_dim) {

    vector<vector<double>> result(x_dim);
    assert(x_dim > 0 && y_dim > 0);
    for (int i = 0; i < x_dim; i++)
        result[i] = create_vector(x_dim);

    return result;
}


vector<double> create_vector(const int& dim) {
    vector<double> result(dim,0);
    for (int i = 0; i < dim; i++) {
        result[i] = (double) rand() / RAND_MAX;
    }
    return result;
}