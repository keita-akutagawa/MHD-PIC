#include <vector>
#include <cmath>
#include <omp.h>
#include <limits>
#include <utility>
#include <iostream>

using namespace std;


void periodic_boudary_condition_x(vector<double>& r,  
                                  int n_start, int n_last, double x_max)
{
    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3){
        if (r[i] > x_max) {
            r[i] = 1e-10;
        } else if (r[i] < 0.0) {
            r[i] = x_max - 1e-10;
        }
    }
}


void periodic_boudary_condition_y(vector<double>& r,  
                                  int n_start, int n_last, double y_max)
{
    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3){
        if (r[i+1] > y_max) {
            r[i+1] = 1e-10;
        } else if (r[i+1] < 0.0) {
            r[i+1] = y_max - 1e-10;
        }
    }
}


void refrective_boudary_condition_x(vector<double>& v, vector<double>& r,  
                                    int n_start, int n_last, double x_min, double x_max)
{
    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3){
        if (r[i] >= x_max) {
            v[i] = -v[i];
            r[i] = 2.0 * x_max - r[i];
        } else if (r[i] <= x_min) {
            v[i] = -v[i];
            r[i] = 2.0 * x_min - r[i];
        }
    }
}


void refrective_boudary_condition_x_left(vector<double>& v, vector<double>& r,  
                                         int n_start, int n_last, double x_min)
{
    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3){
        if (r[i] <= x_min) {
            v[i] = -v[i];
            r[i] = 2.0 * x_min - r[i];
        }
    }
}

void refrective_boudary_condition_x_right(vector<double>& v, vector<double>& r,  
                                         int n_start, int n_last, double x_max)
{
    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3){
        if (r[i] >= x_max) {
            v[i] = -v[i];
            r[i] = 2.0 * x_max - r[i];
        }
    }
}

void refrective_boudary_condition_y(vector<double>& v, vector<double>& r,  
                                    int n_start, int n_last, double y_min, double y_max)
{
    #pragma omp parallel for
    for (int i = 3 * n_start; i < 3 * n_last; i+=3){
        if (r[i+1] >= y_max) {
            v[i+1] = -v[i+1];
            r[i+1] = 2.0 * y_max - r[i+1];
        } else if (r[i+1] <= y_min) {
            v[i+1] = -v[i+1];
            r[i+1] = 2.0 * y_min - r[i+1];
        }
    }
}


void open_boudary_condition_x_right(vector<double>& v, vector<double>& r, vector<double>& r_past, 
                                    vector<double>& in_r, vector<double>& in_v,
                                    vector<double>& out_r, vector<double>& out_v, 
                                    int& in_num, int& out_num, 
                                    int n_start, int n_last, int n_particle, double x_max, double dx)
{
    double inf = numeric_limits<double>::infinity();
    int tmp_in = n_last, tmp_out = 0;

    //#pragma omp parallel for  tmpで衝突するはず
    for (int i = 3 * n_start; i < 3 * n_last; i+=3){
        if (r[i] >= x_max) {
            out_v[3*tmp_out] = v[i];
            out_v[3*tmp_out+1] = v[i+1];
            out_v[3*tmp_out+2] = v[i+2];
            out_r[3*tmp_out] = r[i];
            out_r[3*tmp_out+1] = r[i+1];
            out_r[3*tmp_out+2] = r[i+2];
            v[i] = inf;
            v[i+1] = inf;
            v[i+2] = inf;
            r[i] = inf;
            r[i+1] = inf;
            r[i+2] = inf;
            tmp_out += 1;
        } else if ((r_past[i] > x_max-dx) && (r[i] < x_max-dx)) {
            v[3*tmp_in] = v[i];
            v[3*tmp_in+1] = v[i+1];
            v[3*tmp_in+2] = v[i+2];
            r[3*tmp_in] = r[i] + dx;
            r[3*tmp_in+1] = r[i+1];
            r[3*tmp_in+2] = r[i+2];
            tmp_in += 1;
        }
    }
    in_num = tmp_in - n_last;
    out_num = tmp_out;
}


void sort_buffer(vector<double>& v, vector<double>& r, vector<double>& v_past, vector<double>& r_past, 
                 int& total_ion, int& total_electron, int n_particle)
{
    double inf = numeric_limits<double>::infinity();

    int tmp_ion = 0;
    for (int i = 0; i < 3 * n_particle / 2; i+=3) {
        if (isfinite(r[i])) {
            r_past[3*tmp_ion] = r[i];
            r_past[3*tmp_ion+1] = r[i+1];
            r_past[3*tmp_ion+2] = r[i+2];
            v_past[3*tmp_ion] = v[i];
            v_past[3*tmp_ion+1] = v[i+1];
            v_past[3*tmp_ion+2] = v[i+2];
            tmp_ion += 1;
            if (tmp_ion > n_particle/2) cout << "BROKEN!";
        }
    }
    total_ion = tmp_ion;
    for (int i = 3 * total_ion; i < 3 * n_particle / 2; i+=3) {
        r_past[i] = inf;
        r_past[i+1] = inf;
        r_past[i+2] = inf;
        v_past[i] = inf;
        v_past[i+1] = inf;
        v_past[i+2] = inf;
    }

    int tmp_electron = n_particle / 2;
    for (int i = 3 * n_particle / 2; i < 3 * n_particle; i+=3) {
        if (isfinite(r[i])) {
            r_past[3*tmp_electron] = r[i];
            r_past[3*tmp_electron+1] = r[i+1];
            r_past[3*tmp_electron+2] = r[i+2];
            v_past[3*tmp_electron] = v[i];
            v_past[3*tmp_electron+1] = v[i+1];
            v_past[3*tmp_electron+2] = v[i+2];
            tmp_electron += 1;
            if (tmp_electron > n_particle) cout << "BROKEN!";
        }
    }
    total_electron = tmp_electron - n_particle / 2;
    for (int i = 3 * (n_particle/2 + total_electron); i < 3 * n_particle; i+=3) {
        r_past[i] = inf;
        r_past[i+1] = inf;
        r_past[i+2] = inf;
        v_past[i] = inf;
        v_past[i+1] = inf;
        v_past[i+2] = inf;
    }

    copy(v_past.begin(), v_past.end(), v.begin());
    copy(r_past.begin(), r_past.end(), r.begin());

}


