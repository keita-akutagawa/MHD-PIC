#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>

using namespace std;


void get_particle_field(vector<double>& B_particle, 
                        vector<double>& E_particle, 
                        vector<int>& r_index, 
                        vector<double>& cr,  
                        const vector<vector<vector<double>>> B_tmp, 
                        const vector<vector<vector<double>>> E_tmp, 
                        const vector<double> r, 
                        int n_start, int n_last, 
                        int n_x, int n_y, double dx, double dy)
{
    double cx1cy1, cx1cy2, cx2cy1, cx2cy2;
    int x_index_1, x_index_2, y_index_1, y_index_2;

    #pragma omp parallel for private(cx1cy1, cx1cy2, cx2cy1, cx2cy2, x_index_1, x_index_2, y_index_1, y_index_2)
    for (int i = 3 * n_start; i < 3 * n_last; i+=3) {

        r_index[i] = floor(r[i] / dx);
        r_index[i+1] = floor(r[i+1] / dy);
        cr[i] = r[i] / dx - r_index[i];
        cr[i+1] = r[i+1] / dy - r_index[i+1];

        cx1cy1 = cr[i] * cr[i+1];
        cx1cy2 = cr[i] * (1.0 - cr[i+1]);
        cx2cy1 = (1.0 - cr[i]) * cr[i+1];
        cx2cy2 = (1.0 - cr[i]) * (1.0 - cr[i+1]);

        x_index_1 = r_index[i];
        x_index_2 = x_index_1 + 1;
        y_index_1 = r_index[i+1];
        y_index_2 = y_index_1 + 1;

        if ((x_index_2 != n_x) && (y_index_2 != n_y)) {

            B_particle[i] += B_tmp[0][x_index_1][y_index_1] * cx2cy2;
            B_particle[i] += B_tmp[0][x_index_1][y_index_2] * cx2cy1;
            B_particle[i] += B_tmp[0][x_index_2][y_index_1] * cx1cy2;
            B_particle[i] += B_tmp[0][x_index_2][y_index_2] * cx1cy1;

            B_particle[i+1] += B_tmp[1][x_index_1][y_index_1] * cx2cy2;
            B_particle[i+1] += B_tmp[1][x_index_1][y_index_2] * cx2cy1;
            B_particle[i+1] += B_tmp[1][x_index_2][y_index_1] * cx1cy2;
            B_particle[i+1] += B_tmp[1][x_index_2][y_index_2] * cx1cy1;

            B_particle[i+2] += B_tmp[2][x_index_1][y_index_1] * cx2cy2;
            B_particle[i+2] += B_tmp[2][x_index_1][y_index_2] * cx2cy1;
            B_particle[i+2] += B_tmp[2][x_index_2][y_index_1] * cx1cy2;
            B_particle[i+2] += B_tmp[2][x_index_2][y_index_2] * cx1cy1;

            E_particle[i] += E_tmp[0][x_index_1][y_index_1] * cx2cy2;
            E_particle[i] += E_tmp[0][x_index_1][y_index_2] * cx2cy1;
            E_particle[i] += E_tmp[0][x_index_2][y_index_1] * cx1cy2;
            E_particle[i] += E_tmp[0][x_index_2][y_index_2] * cx1cy1;

            E_particle[i+1] += E_tmp[1][x_index_1][y_index_1] * cx2cy2;
            E_particle[i+1] += E_tmp[1][x_index_1][y_index_2] * cx2cy1;
            E_particle[i+1] += E_tmp[1][x_index_2][y_index_1] * cx1cy2;
            E_particle[i+1] += E_tmp[1][x_index_2][y_index_2] * cx1cy1;

            E_particle[i+2] += E_tmp[2][x_index_1][y_index_1] * cx2cy2;
            E_particle[i+2] += E_tmp[2][x_index_1][y_index_2] * cx2cy1;
            E_particle[i+2] += E_tmp[2][x_index_2][y_index_1] * cx1cy2;
            E_particle[i+2] += E_tmp[2][x_index_2][y_index_2] * cx1cy1;

        } else if (x_index_2 == n_x) {

            B_particle[i] += B_tmp[0][x_index_1][y_index_1] * cx2cy2;
            B_particle[i] += B_tmp[0][x_index_1][y_index_2] * cx2cy1;
            //B_particle[i] += B_tmp[0][0][y_index_1] * cx1cy2;
            //B_particle[i] += B_tmp[0][0][y_index_2] * cx1cy1;

            B_particle[i+1] += B_tmp[1][x_index_1][y_index_1] * cx2cy2;
            B_particle[i+1] += B_tmp[1][x_index_1][y_index_2] * cx2cy1;
            //B_particle[i+1] += B_tmp[1][0][y_index_1] * cx1cy2;
            //B_particle[i+1] += B_tmp[1][0][y_index_2] * cx1cy1;

            B_particle[i+2] += B_tmp[2][x_index_1][y_index_1] * cx2cy2;
            B_particle[i+2] += B_tmp[2][x_index_1][y_index_2] * cx2cy1;
            //B_particle[i+2] += B_tmp[2][0][y_index_1] * cx1cy2;
            //B_particle[i+2] += B_tmp[2][0][y_index_2] * cx1cy1;

            E_particle[i] += E_tmp[0][x_index_1][y_index_1] * cx2cy2;
            E_particle[i] += E_tmp[0][x_index_1][y_index_2] * cx2cy1;
            //E_particle[i] += E_tmp[0][0][y_index_1] * cx1cy2;
            //E_particle[i] += E_tmp[0][0][y_index_2] * cx1cy1;

            E_particle[i+1] += E_tmp[1][x_index_1][y_index_1] * cx2cy2;
            E_particle[i+1] += E_tmp[1][x_index_1][y_index_2] * cx2cy1;
            //E_particle[i+1] += E_tmp[1][0][y_index_1] * cx1cy2;
            //E_particle[i+1] += E_tmp[1][0][y_index_2] * cx1cy1;

            E_particle[i+2] += E_tmp[2][x_index_1][y_index_1] * cx2cy2;
            E_particle[i+2] += E_tmp[2][x_index_1][y_index_2] * cx2cy1;
            //E_particle[i+2] += E_tmp[2][0][y_index_1] * cx1cy2;
            //E_particle[i+2] += E_tmp[2][0][y_index_2] * cx1cy1;

        } else if (y_index_2 == n_y) {

            B_particle[i] += B_tmp[0][x_index_1][y_index_1] * cx2cy2;
            //B_particle[i] += B_tmp[0][x_index_1][0] * cx2cy1;
            B_particle[i] += B_tmp[0][x_index_2][y_index_1] * cx1cy2;
            //B_particle[i] += B_tmp[0][x_index_2][0] * cx1cy1;

            B_particle[i+1] += B_tmp[1][x_index_1][y_index_1] * cx2cy2;
            //B_particle[i+1] += B_tmp[1][x_index_1][0] * cx2cy1;
            B_particle[i+1] += B_tmp[1][x_index_2][y_index_1] * cx1cy2;
            //B_particle[i+1] += B_tmp[1][x_index_2][0] * cx1cy1;

            B_particle[i+2] += B_tmp[2][x_index_1][y_index_1] * cx2cy2;
            //B_particle[i+2] += B_tmp[2][x_index_1][0] * cx2cy1;
            B_particle[i+2] += B_tmp[2][x_index_2][y_index_1] * cx1cy2;
            //B_particle[i+2] += B_tmp[2][x_index_2][0] * cx1cy1;

            E_particle[i] += E_tmp[0][x_index_1][y_index_1] * cx2cy2;
            //E_particle[i] += E_tmp[0][x_index_1][0] * cx2cy1;
            E_particle[i] += E_tmp[0][x_index_2][y_index_1] * cx1cy2;
            //E_particle[i] += E_tmp[0][x_index_2][0] * cx1cy1;

            E_particle[i+1] += E_tmp[1][x_index_1][y_index_1] * cx2cy2;
            //E_particle[i+1] += E_tmp[1][x_index_1][0] * cx2cy1;
            E_particle[i+1] += E_tmp[1][x_index_2][y_index_1] * cx1cy2;
            //E_particle[i+1] += E_tmp[1][x_index_2][0] * cx1cy1;

            E_particle[i+2] += E_tmp[2][x_index_1][y_index_1] * cx2cy2;
            //E_particle[i+2] += E_tmp[2][x_index_1][0] * cx2cy1;
            E_particle[i+2] += E_tmp[2][x_index_2][y_index_1] * cx1cy2;
            //E_particle[i+2] += E_tmp[2][x_index_2][0] * cx1cy1;

        } else {

            B_particle[i] += B_tmp[0][x_index_1][y_index_1] * cx2cy2;
            //B_particle[i] += B_tmp[0][x_index_1][0] * cx2cy1;
            //B_particle[i] += B_tmp[0][0][y_index_1] * cx1cy2;
            //B_particle[i] += B_tmp[0][0][0] * cx1cy1;

            B_particle[i+1] += B_tmp[1][x_index_1][y_index_1] * cx2cy2;
            //B_particle[i+1] += B_tmp[1][x_index_1][0] * cx2cy1;
            //B_particle[i+1] += B_tmp[1][0][y_index_1] * cx1cy2;
            //B_particle[i+1] += B_tmp[1][0][0] * cx1cy1;

            B_particle[i+2] += B_tmp[2][x_index_1][y_index_1] * cx2cy2;
            //B_particle[i+2] += B_tmp[2][x_index_1][0] * cx2cy1;
            //B_particle[i+2] += B_tmp[2][0][y_index_1] * cx1cy2;
            //B_particle[i+2] += B_tmp[2][0][0] * cx1cy1;

            E_particle[i] += E_tmp[0][x_index_1][y_index_1] * cx2cy2;
            //E_particle[i] += E_tmp[0][x_index_1][0] * cx2cy1;
            //E_particle[i] += E_tmp[0][0][y_index_1] * cx1cy2;
            //E_particle[i] += E_tmp[0][0][0] * cx1cy1;

            E_particle[i+1] += E_tmp[1][x_index_1][y_index_1] * cx2cy2;
            //E_particle[i+1] += E_tmp[1][x_index_1][0] * cx2cy1;
            //E_particle[i+1] += E_tmp[1][0][y_index_1] * cx1cy2;
            //E_particle[i+1] += E_tmp[1][0][0] * cx1cy1;

            E_particle[i+2] += E_tmp[2][x_index_1][y_index_1] * cx2cy2;
            //E_particle[i+2] += E_tmp[2][x_index_1][0] * cx2cy1;
            //E_particle[i+2] += E_tmp[2][0][y_index_1] * cx1cy2;
            //E_particle[i+2] += E_tmp[2][0][0] * cx1cy1;

        }
    }
}

