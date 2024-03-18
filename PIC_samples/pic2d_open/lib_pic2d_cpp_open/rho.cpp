#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>

using namespace std;


void get_rho(vector<vector<double>>& rho, 
             vector<int>& r_index, 
             vector<double>& cr, 
             const vector<double> r, 
             int n_start, int n_last, double q, 
             int n_x, int n_y, double dx, double dy)
{

    double cx1cy1, cx1cy2, cx2cy1, cx2cy2;
    int x_index_1, x_index_2, y_index_1, y_index_2;

    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        r_index[i] = floor(r[i] / dx);
        r_index[i+1] = floor(r[i+1] / dy);
        cr[i] = r[i] / dx - r_index[i];
        cr[i+1] = r[i+1] / dy - r_index[i+1];
    }

    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {

        cx1cy1 = cr[i] * cr[i+1];
        cx1cy2 = cr[i] * (1.0 - cr[i+1]);
        cx2cy1 = (1.0 - cr[i]) * cr[i+1];
        cx2cy2 = (1.0 - cr[i]) * (1.0 - cr[i+1]);

        x_index_1 = r_index[i];
        x_index_2 = x_index_1 + 1;
        y_index_1 = r_index[i+1];
        y_index_2 = y_index_1 + 1;

        if ((x_index_2 != n_x) && (y_index_2 != n_y)) {

            rho[x_index_1][y_index_1] += q * cx2cy2;
            rho[x_index_1][y_index_2] += q * cx2cy1;
            rho[x_index_2][y_index_1] += q * cx1cy2;
            rho[x_index_2][y_index_2] += q * cx1cy1;

        } else if (x_index_2 == n_x) {
            
            rho[x_index_1][y_index_1] += q * cx2cy2;
            rho[x_index_1][y_index_2] += q * cx2cy1;
            //rho[0][y_index_1] += q * cx1cy2;
            //rho[0][y_index_2] += q * cx1cy1;

        } else if (y_index_2 == n_y) {

            rho[x_index_1][y_index_1] += q * cx2cy2;
            //rho[x_index_1][0] += q * cx2cy1;
            rho[x_index_2][y_index_1] += q * cx1cy2;
            //rho[x_index_2][0] += q * cx1cy1;

        } else {

            rho[x_index_1][y_index_1] += q * cx2cy2;
            //rho[x_index_1][0] += q * cx2cy1;
            //rho[0][y_index_1] += q * cx1cy2;
            //rho[0][0] += q * cx1cy1;

        }
    }
    
}


void get_rho_open(vector<vector<double>>& rho, 
             vector<int>& cross_r_index, 
             vector<double>& cross_cr,  
             const vector<double> cross_r,
             int n_start, int n_last, double q, 
             int n_x, int n_y, double dx, double dy)
{

    double cx1cy1, cx1cy2, cx2cy1, cx2cy2;
    int x_index_1, x_index_2, y_index_1, y_index_2;

    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {
        cross_r_index[i] = floor(cross_r[i] / dx);
        cross_r_index[i+1] = floor(cross_r[i+1] / dy);
        cross_cr[i] = cross_r[i] / dx - cross_r_index[i];
        cross_cr[i+1] = cross_r[i+1] / dy - cross_r_index[i+1];
    }

    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {

        cx1cy1 = cross_cr[i] * cross_cr[i+1];
        cx1cy2 = cross_cr[i] * (1.0 - cross_cr[i+1]);
        cx2cy1 = (1.0 - cross_cr[i]) * cross_cr[i+1];
        cx2cy2 = (1.0 - cross_cr[i]) * (1.0 - cross_cr[i+1]);

        y_index_1 = cross_r_index[i+1];
        y_index_2 = y_index_1 + 1;

        if (y_index_2 != n_y) {

            rho[n_x-1][y_index_1] += q * cx2cy2;
            rho[n_x-1][y_index_2] += q * cx2cy1;

        } else {

            rho[n_x-1][y_index_1] += q * cx2cy2;
            rho[n_x-1][0] += q * cx2cy1;

        }
    }
}


