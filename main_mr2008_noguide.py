import os 
import sys
sys.path.append(os.pardir)
import numpy as np

from lib_pic2d_numba.boundary import *
from lib_pic2d_numba.particles_booster import *
from lib_pic2d_numba.maxwell import *
from lib_pic2d_numba.others import *
from const_mr2008_noguide import *



current_path = os.getcwd()
if not os.path.exists(current_path + '/results_mr2008_noguide'):
    os.mkdir('results_mr2008_noguide')

filename = 'progress.txt'
f = open(filename, 'w')



m_list, q_list, v, x = sort_list(m_list, q_list, v, x, dx, dy, n_x, n_y)
#STEP1
rho = get_rho(q_list, x, n_x, n_y, dx, dy)
E = solve_poisson_not_periodic(rho, n_x, n_y, dx, dy, epsilon0, E)    

for k in range(step+1):
    if k == 0:
        B += delta_B
    #STEP2
    B = time_evolution_B(E, dx, dy, dt/2, B)
    B[1, :, [0, 1]] = 0.0
    B[1, :, -1] = 0.0
    B[0, :, 0] = -mu_0 * current[2, :, 0]
    B[2, :, 0] = mu_0 * current[0, :, 0]
    B[0, :, -1] = mu_0 * current[2, :, -1]
    B[2, :, -1] = -mu_0 * current[0, :, -1]
    B[1, 0, :] = 0.0
    B[1, [-1, -2], :] = 0.0
    B[2, 0, :] = 0.0
    B[2, [-1, -2], :] = 0.0
    B[0, [0, 1], :] = B[0, [1, 2], :]
    B[0, [-1, -2], :] = B[0, [-2, -3], :]
    #STEP3
    E_tmp = E.copy()
    B_tmp = B.copy()
    #整数格子点上に再定義。特に磁場は平均の取り方に注意。
    E_tmp[0, :, :] = (E[0, :, :] + np.roll(E[0, :, :], 1, axis=0)) / 2.0
    E_tmp[1, :, :] = (E[1, :, :] + np.roll(E[1, :, :], 1, axis=1)) / 2.0
    B_tmp[0, :, :] = (B[0, :, :] + np.roll(B[0, :, :], 1, axis=1)) / 2.0
    B_tmp[1, :, :] = (B[1, :, :] + np.roll(B[1, :, :], 1, axis=0)) / 2.0
    B_tmp[2, :, :] = (B[2, :, :] + np.roll(B[2, :, :], 1, axis=0) + np.roll(B[2, :, :], 1, axis=1) + np.roll(B[2, :, :], [1, 1], axis=[0, 1])) / 4.0
    v = time_evolution_v(c, E_tmp, B_tmp, x, q_list, m_list, n_x, n_y, dx, dy, dt, v)
    #STEP4
    x = time_evolution_x(c, dt/2, v, x)
    v, x = refrective_condition_x(v, x, x_max)
    v, x = refrective_condition_y(v, x, y_max)
    #STEP5
    current = get_current_density(c, q_list, v, x, n_x, n_y, dx, dy, current)
    current[0, :, :] = (current[0, :, :] + np.roll(current[0, :, :], -1, axis=0)) / 2.0
    current[1, :, :] = (current[1, :, :] + np.roll(current[1, :, :], -1, axis=1)) / 2.0
    #STEP6
    B = time_evolution_B(E, dx, dy, dt/2, B)
    B[1, :, [0, 1]] = 0.0
    B[1, :, -1] = 0.0
    B[0, :, 0] = -mu_0 * current[2, :, 0]
    B[2, :, 0] = mu_0 * current[0, :, 0]
    B[0, :, -1] = mu_0 * current[2, :, -1]
    B[2, :, -1] = -mu_0 * current[0, :, -1]
    B[1, 0, :] = 0.0
    B[1, [-1, -2], :] = 0.0
    B[2, 0, :] = 0.0
    B[2, [-1, -2], :] = 0.0
    B[0, [0, 1], :] = B[0, [1, 2], :]
    B[0, [-1, -2], :] = B[0, [-2, -3], :]
    #STEP7
    if k % 10 == 0:
        with open(filename, 'a') as f:
            f.write(f'{int(k*dt)} step done...\n')
    E = time_evolution_E(B, current, c, epsilon0, dx, dy, dt, E)
    rho = get_rho(q_list, x, n_x, n_y, dx, dy)
    E[1, :, 0] = rho[:, 0] / epsilon0
    E[1, :, -1] = -rho[:, -1] / epsilon0
    E[0, :, [0, 1]] = 0.0
    E[2, :, [0, 1]] = 0.0
    E[0, :, -1] = 0.0
    E[2, :, -1] = 0.0
    E[0, 0, :] = 0.0
    E[0, [-1, -2], :] = 0.0
    E[1, [0, 1], :] = E[1, [1, 2], :]
    E[1, [-1, -2], :] = E[1, [-2, -3], :]
    E[2, [0, 1], :] = E[2, [1, 2], :]
    E[2, [-1, -2], :] = E[2, [-2, -3], :]
    #STEP8
    x = time_evolution_x(c, dt/2, v, x)
    v, x = refrective_condition_x(v, x, x_max)
    v, x = refrective_condition_y(v, x, y_max)

    if k % 1000 == 0:
        k1 = k // 1000
        KE = np.sum(1/2 * m_list * np.linalg.norm(v, axis=0)**2)
        np.save(f'results_mr2008_noguide/results_mr2008_xv_{k1}.npy', np.concatenate([x, v]))
        np.save(f'results_mr2008_noguide/results_mr2008_E_{k1}.npy', E)
        np.save(f'results_mr2008_noguide/results_mr2008_B_{k1}.npy', B)
        np.save(f'results_mr2008_noguide/results_mr2008_current_{k1}.npy', current)
        np.save(f'results_mr2008_noguide/results_mr2008_KE_{k1}.npy', KE)



sys.exit()


