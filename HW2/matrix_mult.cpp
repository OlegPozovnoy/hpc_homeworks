//
// Created by oleg on 06.10.2020.
//

#include "main.h"
#include <cassert>

vector<vector<double>> matrix_mult(const vector<vector<double>> A, const vector<vector<double>> B) {
    assert(A[0].size() == B.size());
    vector<vector<double>> result(B.size());

    for (size_t i = 0; i < B.size(); i++) {
        result[i] = vect_matrix_mult(A[i], B);
    }

    return result;
}


vector<double> vect_matrix_mult(const vector<double> a, const vector<vector<double>> B) {
    vector<double> result(B[0].size(), 0);
    assert(a.size() == B.size());
    for (size_t i = 0; i < B.size(); i++) {
        double next = 0;
        for (size_t j = 0; j < a.size(); j++) {
            next += a[j] * B[j][i];
        }
        result[i] = next;
    }
    return result;
}