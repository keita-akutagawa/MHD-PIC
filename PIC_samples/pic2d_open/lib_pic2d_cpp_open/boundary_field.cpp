#include <vector>
#include <cmath>
#include <omp.h>
#include <limits>
#include <utility>
#include <iostream>

using namespace std;


void boundary_B(vector<vector<vector<double>>>& B, 
                int n_x, int n_y, double dx, double dy, double B0_g)
{

    for (int j = 0; j < n_y; j++) {
        B[0][0][j] = B[0][1][j];
        B[1][0][j] = 0.0;
        B[1][n_x-1][j] = B[1][n_x-2][j];
        B[2][0][j] = B0_g;
        B[2][n_x-1][j] = B[2][n_x-2][j];
    }
    for (int j = 0; j < n_y-1; j++) {
        B[0][n_x-1][j] = -(B[1][n_x-1][(j+1)%n_y] - B[1][n_x-1][j]) / dy * dx
                       + B[0][n_x-2][j];
    } //j = n_y-1は導体壁の境界条件の方でカバーする。

    for (int i = 0; i < n_x; i++) {
        B[1][i][0] = 0.0;
        B[1][i][1] = 0.0;
        B[1][i][n_y-1] = 0.0;
        B[2][i][0] = B[2][i][1];
        B[2][i][n_y-1] = B[2][i][n_y-2];
    }
    for (int i = 0; i < n_x-1; i++) {
        B[0][i+1][0] = -(B[1][i][1] - B[1][i][0]) / dy * dx + B[0][i][0];
        B[0][i+1][n_y-2] = -(B[1][i][n_y-1] - B[1][i][n_y-2]) / dy * dx + B[0][i][n_y-2];
        B[0][i+1][n_y-1] = -(0.0 - B[1][i][n_y-1]) / dy * dx + B[0][i][n_y-1];
    }

}


void boundary_E(vector<vector<vector<double>>>& E, vector<vector<double>> rho, 
                int n_x, int n_y, double dx, double dy, double epsilon0)
{

    for (int j = 0; j < n_y; j++) {
        E[0][0][j] = 0.0;
        E[1][0][j] = E[1][1][j];
        E[2][0][j] = E[2][1][j];
        E[1][n_x-1][j] = E[1][n_x-2][j];
        E[2][n_x-1][j] = E[2][n_x-2][j];
    }
    for (int j = 1; j < n_y; j++) {
        E[0][n_x-1][j] = (-(E[1][n_x-1][j] - E[1][n_x-1][j-1])/dy + rho[n_x-1][j]/epsilon0) * dx 
                       + E[0][n_x-2][j];
    }

    for (int i = 0; i < n_x; i++) {
        E[0][i][0] = 0.0;
        E[0][i][1] = 0.0;
        E[0][i][n_y-1] = 0.0;
        E[1][i][0] = rho[i][0] / epsilon0;
        E[1][i][n_y-1] = -(rho[i][n_y-1] + 0.0) / 2.0 / epsilon0;
        E[2][i][0] = 0.0;
        E[2][i][1] = 0.0;
        E[2][i][n_y-1] = 0.0;
    }
}