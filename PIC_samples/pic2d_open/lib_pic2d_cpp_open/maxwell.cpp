#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>

using namespace std;


void time_evolution_B(vector<vector<vector<double>>>& B, 
                      const vector<vector<vector<double>>> E, 
                      int n_x, int n_y, double dx, double dy, double dt) 
{
    int i, j = 0;
    #pragma omp parallel for private(j)
    for (i = 0; i < n_x; i++) {
        for (j = 0; j < n_y; j++) {
            B[0][i][j] += -(E[2][i][(j+1)%n_y] - E[2][i][j]) / dy * dt;
            B[1][i][j] += (E[2][(i+1)%n_x][j] - E[2][i][j]) / dx * dt;
            B[2][i][j] += (-(E[1][(i+1)%n_x][j] - E[1][i][j]) / dx
                        + (E[0][i][(j+1)%n_y] - E[0][i][j]) / dy) * dt;
        }
    }
}


void time_evolution_E(vector<vector<vector<double>>>& E, 
                      const vector<vector<vector<double>>> B, 
                      const vector<vector<vector<double>>> current,
                      int n_x, int n_y, double dx, double dy, double dt, 
                      double c, double epsilon0) 
{
    int i, j = 0;
    #pragma omp parallel for private(j)
    for (i = 0; i < n_x; i++) {
        for (j = 0; j < n_y; j++) {
            E[0][i][j] += (-current[0][i][j] / epsilon0
                        + pow(c, 2) * (B[2][i][j] - B[2][i][(n_y+j-1)%n_y]) / dy) * dt;
            E[1][i][j] += (-current[1][i][j] / epsilon0 
                        - pow(c, 2) * (B[2][i][j] - B[2][(n_x+i-1)%n_x][j]) / dx) * dt;
            E[2][i][j] += (-current[2][i][j] / epsilon0 
                        + pow(c, 2) * ((B[1][i][j] - B[1][(n_x+i-1)%n_x][j]) / dx
                        - (B[0][i][j] - B[0][i][(n_y+j-1)%n_y]) / dy)) * dt;
        }
    }
}
