import numpy as np
from scipy import stats


c = 0.5
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1/9
m_electron = 1 * m_unit
m_ion = m_electron / r_m
t_r = 1.0
r_q = 1.0
n_e = 10 #ここは手動で調整すること
B0 = np.sqrt(n_e) / 1.5
n_i = int(n_e / r_q)
T_i  = (B0**2 / 2.0 / mu_0) / (n_i + n_e * t_r)
T_e = T_i * t_r
q_unit = np.sqrt(epsilon0 * T_e / n_e)
q_electron = -1 * q_unit
q_ion = r_q * q_unit
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
ion_inertial_length = c / omega_pi
sheat_thickness = 1.0 * ion_inertial_length
v_electron = np.array([0.0, 0.0, c * debye_length / sheat_thickness * np.sqrt(2 / (1.0 + 1/t_r))])
v_ion = -v_electron / t_r
v_thermal_electron = np.sqrt(T_e / m_electron)
v_thermal_ion = np.sqrt(T_i / m_ion)

dx = 1.0
dy = 1.0
n_x = int(ion_inertial_length * 200)
n_y = int(ion_inertial_length * 50)
x_max = n_x * dx
y_max = n_y * dy
x_coordinate = np.arange(0.0, x_max, dx)
y_coordinate = np.arange(0.0, y_max, dy)
dt = 1.0
step = 20000
t_max = step * dt


E = np.zeros([3, n_x, n_y])
B = np.zeros([3, n_x, n_y])
current = np.zeros([3, n_x, n_y])
for j in range(n_y):
    B[0, :, j] = B0 * np.tanh((y_coordinate[j] - y_max/2) / sheat_thickness)



reconnection_ratio = 0.1
Xpoint_position = 10 * ion_inertial_length
delta_B = np.zeros([3, n_x, n_y])
X, Y = np.meshgrid(x_coordinate, y_coordinate)
delta_B[0, :, :] = -np.array(reconnection_ratio * B0 * (Y - y_max/2) / sheat_thickness \
                 * np.exp(-((X - Xpoint_position)**2 + (Y - y_max/2)**2) / ((2.0 * sheat_thickness)**2))).T
delta_B[1, :, :] = np.array(reconnection_ratio * B0 * (X - Xpoint_position) / sheat_thickness \
                 * np.exp(-((X - Xpoint_position)**2 + (Y - y_max/2)**2) / ((2.0 * sheat_thickness)**2))).T

n_ion = int(n_x * n_i * 2.0 * sheat_thickness)
n_electron = int(n_ion * abs(q_ion / q_electron))
n_ion_background = int(n_x * 0.2 * n_i * (y_max - 2.0 * sheat_thickness))
n_electron_background = int(n_x * 0.2 * n_e * (y_max - 2.0 * sheat_thickness))
n_particle = n_ion + n_ion_background + n_electron + n_electron_background
x = np.zeros([3, n_particle])
v = np.zeros([3, n_particle])
print(f"total number of particles is {n_particle}.")

np.random.RandomState(1)
x_start_plus = np.random.rand(n_ion) * x_max
x_start_plus_background = np.random.rand(n_ion_background) * x_max
x_start_minus = np.random.rand(n_electron) * x_max
x_start_minus_background = np.random.rand(n_electron_background) * x_max
y_start_plus = np.array(y_max/2 + sheat_thickness * np.arctanh(2.0 * np.random.rand(n_ion) - 1.0))
y_start_plus[y_start_plus > y_max] = y_max/2
y_start_plus[y_start_plus < 0.0] = y_max/2
y_start_plus_background = np.zeros(n_ion_background)
for i in range(n_ion_background):
    while True:
        rand = np.random.rand(1) * y_max 
        rand_pn = np.random.rand(1)
        if rand_pn < (1.0 - 1.0/np.cosh((rand - y_max/2)/sheat_thickness)):
            y_start_plus_background[i] = rand
            break
y_start_minus = np.array(y_max/2 + sheat_thickness * np.arctanh(2.0 * np.random.rand(n_electron) - 1.0))
y_start_minus[y_start_minus > y_max] = y_max/2
y_start_minus[y_start_minus < 0.0] = y_max/2
y_start_minus_background = np.zeros(n_electron_background)
for i in range(n_electron_background):
    while True:
        rand = np.random.rand(1) * y_max 
        rand_pn = np.random.rand(1)
        if rand_pn < (1.0 - 1.0/np.cosh((rand - y_max/2)/sheat_thickness)):
            y_start_minus_background[i] = rand
            break
x_start_plus = np.array(x_start_plus)
x_start_plus_background = np.array(x_start_plus_background)
x_start_minus = np.array(x_start_minus)
x_start_minus_background = np.array(x_start_minus_background)
y_start_plus = np.array(y_start_plus)
y_start_plus_background = np.array(y_start_plus_background)
y_start_minus = np.array(y_start_minus)
y_start_minus_background = np.array(y_start_minus_background)
x[0, :] = np.concatenate([x_start_plus, x_start_plus_background, x_start_minus, x_start_minus_background])
x[1, :] = np.concatenate([y_start_plus, y_start_plus_background, y_start_minus, y_start_minus_background])
v[0, :n_ion] = np.array(stats.norm.rvs(v_ion[0], v_thermal_ion, size=n_ion))
v[0, n_ion:n_ion+n_ion_background] = np.array(stats.norm.rvs(0.0, v_thermal_ion, size=n_ion_background))
v[0, n_ion+n_ion_background:n_ion+n_ion_background+n_electron] = np.array(stats.norm.rvs(v_electron[0], v_thermal_electron, size=n_electron))
v[0, n_ion+n_ion_background+n_electron:] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_electron_background))
v[1, :n_ion] = np.array(stats.norm.rvs(v_ion[1], v_thermal_ion, size=n_ion))
v[1, n_ion:n_ion+n_ion_background] = np.array(stats.norm.rvs(0.0, v_thermal_ion, size=n_ion_background))
v[1, n_ion+n_ion_background:n_ion+n_ion_background+n_electron] = np.array(stats.norm.rvs(v_electron[1], v_thermal_electron, size=n_electron))
v[1, n_ion+n_ion_background+n_electron:] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_electron_background))
v[2, :n_ion] = np.array(stats.norm.rvs(v_ion[2], v_thermal_ion, size=n_ion))
v[2, n_ion:n_ion+n_ion_background] = np.array(stats.norm.rvs(0.0, v_thermal_ion, size=n_ion_background))
v[2, n_ion+n_ion_background:n_ion+n_ion_background+n_electron] = np.array(stats.norm.rvs(v_electron[2], v_thermal_electron, size=n_electron))
v[2, n_ion+n_ion_background+n_electron:] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_electron_background))

q_list = np.zeros(n_particle)
q_list[:n_ion+n_ion_background] = q_ion
q_list[n_ion+n_ion_background:] = q_electron
m_list = np.zeros(n_particle)
m_list[:n_ion+n_ion_background] = m_ion
m_list[n_ion+n_ion_background:] = m_electron