#include<iostream>
#include<vector>
#include<ctime>
#include "main.h"

using std::vector;
using std::cout;
using std::cin;

void test_mult(int dim);

int main() {
    test_mult(512);
    test_mult(1024);
    test_mult(2048);
    test_mult(4096);
    return 0;
}

void test_mult(int dim) {
    clock_t start = clock();

    vector<vector<double>> A = create_matrix(dim, dim);
    vector<vector<double>> B = create_matrix(dim, dim);

    matrix_mult(A, B);

    cout << "Running time for dim="<<dim<<": " << (double) (clock() - start) / CLOCKS_PER_SEC << "\n";
}
