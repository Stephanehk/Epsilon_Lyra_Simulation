import numpy as np
from numpy import linalg as LA
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

G = 6.67408e-11
#sun mass
sun_mass = 1.989e+30
earth_mass = 5.972e24
#velocity of earth around the sun
earth_sun_v =30000
#fix later
orbital_period_1 = 79.91*365*24*3600*0.51 #in seconds
distance_between_stars1 = 5.326e+12
K1 = (G*orbital_period_1*sun_mass)/(np.power(distance_between_stars1,2) * earth_sun_v)
K2 = (earth_sun_v*orbital_period_1)/distance_between_stars1

def calculate_2_body (initial_conditions,t,m1,m2):
    r1 = initial_conditions[0:3]
    r2 = initial_conditions[3:6]
    v1 = initial_conditions[6:9]
    v2 = initial_conditions[9:12]
    e = 0.8
    G = 6.67408e-11


    #calculate 2 body equation
    v_1_after = K1*(m2*(r2-r1))/(np.power(LA.norm(r2 - r1), 3))
    v_2_after = K1*(m1*(r1-r2))/np.power(LA.norm(r2 - r1), 3)

    r_1_after =K2*v1
    r_2_after =K2*v2

    #make sure eveyrthing is in the right format WTF GOING ON
    v_1_after = np.array(v_1_after, dtype="float64")
    v_2_after = np.array(v_2_after, dtype="float64")
    r_1_after = np.array(r_1_after, dtype="float64")
    r_2_after = np.array(r_2_after, dtype="float64")

    #concatenate everything into big happy array
    r_derivs = scipy.concatenate((r_1_after, r_2_after))
    all_derivs = scipy.concatenate((r_derivs, v_1_after, v_2_after))
    return all_derivs

def main (m1,m2,r1,r2):
    m1 = m1/earth_mass
    m2 = m2/earth_mass
    e = 0.8
    G = 6.67408e-11
    #find position of center of mass
    r_com = ((m1*r1) + (m2*r2))/(m1+m2)

    #TODO Calculate intial velocity as vector not scalar or predefined vector
    #find intiial velocity
    v_1_initial = [0.01,0.01,0]
    #print ((G*m2*(1+e)*LA.norm(r_com - r1))/(np.power(LA.norm(r2 - r1),2)))
    #[0.01,0.01,0]
    v_2_initial = [-0.05,0,-0.1]
    #(G*m1*(1+e)*LA.norm(r_com - r2))/(np.power(LA.norm(r2 - r1),2))
    #[-0.05,0,-0.1]

    initial_conditions = scipy.array([r1,r2, v_1_initial, v_2_initial]).flatten()

    #0, number of orbital periods, number of points to calculate
    t = np.linspace(0,8,500)
    #intergate
    res = integrate.odeint(calculate_2_body, initial_conditions, t, args=(m1,m2))
    r_1_pos = res[:,:3]
    r_2_pos = res[:,3:6]

    #---------------------------------PLOT------------------------------------------------------------
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection="3d")

    #UNCOMMENT TO Plot from COM:
    r_com = ((m1*r_1_pos) +(m2*r_2_pos))/(m1+m2)
    r_1_pos = r_1_pos - r_com
    r_2_pos = r_2_pos - r_com

    ax.plot(r_1_pos[:,0],r_1_pos[:,1],r_1_pos[:,2],color="darkblue")
    ax.plot(r_2_pos[:,0],r_2_pos[:,1],r_2_pos[:,2],color="tab:red")
    ax.scatter(r_1_pos[-1,0],r_1_pos[-1,1],r_1_pos[-1,2],color="darkblue",marker="o",s=100,label="Star1")
    ax.scatter(r_2_pos[-1,0],r_2_pos[-1,1],r_2_pos[-1,2],color="tab:red",marker="o",s=100,label="Star2")
    plt.show()

main (7.3e24,6.8e24,np.array([-0.5,0,0],dtype="float64"),np.array([0.5,0,0],dtype="float64"))
