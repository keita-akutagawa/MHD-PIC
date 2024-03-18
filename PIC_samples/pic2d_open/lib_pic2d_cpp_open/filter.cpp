#include <vector>
#include <cmath>
#include <omp.h>
#include <limits>
#include <utility>
#include <iostream>

using namespace std;


void filter_E(vector<vector<vector<double>>>& E, 
              vector<vector<double>> rho, vector<vector<double>>& F, 
              int n_x, int n_y, double dx, double dy, double dt, 
              double epsilon0, double dmin, double dmax)
{
    for (int i = 1; i < n_x; i++) {
        for (int j = 1; j < n_y; j++) {
            F[i][j] = ((E[0][i][j] - E[0][(i-1+n_x)%n_x][j])/dx + (E[1][i][j] - E[1][i][(j-1+n_y)%n_y])/dy)
                    - rho[i][j] / epsilon0;
        }
    }
    for (int j = 0; j < n_y; j++) {
        F[0][j] = 0.0;
    }
    for (int i = 1; i < n_x; i++) {
        F[i][0] = 0.0;
    }

    double x_max = n_x * dx;
    double d_dx = -(dmax - dmin) / x_max * 100.0;
    double d1, d2;

    for (int i = 0; i < n_x-1; i++) {
        for (int j = 0; j < n_y-1; j++) {
            d1 = min(dmax, dmax + d_dx * ((i + 1) * dx - 99.0/100.0 * x_max));
            d2 = min(dmax, dmax + d_dx * (i * dx - 99.0/100.0 * x_max));
            E[0][i][j] += dt * (d1 * F[(i+1)%n_x][j] - d2 * F[i][j]) / dx;
            E[1][i][j] += dt * (d1 * F[i][(j+1)%n_y] - d2 * F[i][j]) / dy;
        }
    }

    //端は電荷密度同じにしておく。
    for (int i = 0; i < n_x-1; i++) {
        d1 = min(dmax, dmax + d_dx * ((i + 1) * dx - 99.0/100.0 * x_max));
        d2 = min(dmax, dmax + d_dx * (i * dx - 99.0/100.0 * x_max));
        E[0][i][n_y-1] += dt * (d1 * F[(i+1)%n_x][n_y-1] - d2 * F[i][n_y-1]) / dx;
        E[1][i][n_y-1] += dt * (d1 * 0.0 - d2 * F[i][n_y-1]) / dy;
    }

    for (int j = 0; j < n_y-1; j++) {
        d1 = min(dmax, dmax + d_dx * (n_x * dx - 99.0/100.0 * x_max));
        d2 = min(dmax, dmax + d_dx * ((n_x-1) * dx - 99.0/100.0 * x_max));
        E[0][n_x-1][j] += dt * (d1 * 0.0 - d2 * F[n_x-1][j]) / dx; //多分大丈夫。
        E[1][n_x-1][j] += dt * (d1 * F[n_x-1][(j+1)%n_y] - d2 * F[n_x-1][j]) / dy;
    }

    d1 = min(dmax, dmax + d_dx * (n_x * dx - 99.0/100.0 * x_max));
    d2 = min(dmax, dmax + d_dx * ((n_x-1) * dx - 99.0/100.0 * x_max));
    E[0][n_x-1][n_y-1] += dt * dmin * (d1 * 0.0 - d2 * F[n_x-1][n_y-1])/dx; //多分大丈夫。
    E[1][n_x-1][n_y-1] += dt * dmin * (d1 * F[n_x-1][n_y-1] - d2 * F[n_x-1][n_y-2])/dy; //多分大丈夫。

}