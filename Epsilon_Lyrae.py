import numpy as np
from numpy import linalg as LA
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#------------------------------BIG LIST OF CONSTANTS--------------------------------------------
G = 6.67408e-11
#sun mass
sun_mass = 1.989e+30
earth_mass = 5.972e24
#velocity of earth around the sun
earth_sun_v =30000
#--------------------------------------------------------------------------------------------
orbital_period_1 = 5.6903874e+10 #AB PERIOD
#79.91*365*24*3600*0.51 #in seconds
distance_between_stars1 = 17400000000
#5.326e+12
K1_1 = (G*orbital_period_1*sun_mass)/(np.power(distance_between_stars1,2) * earth_sun_v)
K2_1 = (earth_sun_v*orbital_period_1)/distance_between_stars1
#--------------------------------------------------------------------------------------------
orbital_period_2 = 2.2841746e+10 #CD PERIOD
#79.91*365*24*3600*0.51 #in seconds
distance_between_stars2 = 18150000000
#5.326e+12
K1_2 = (G*orbital_period_2*sun_mass)/(np.power(distance_between_stars2,2) * earth_sun_v)
K2_2 = (earth_sun_v*orbital_period_2)/distance_between_stars2
#--------------------------------------------------------------------------------------------
def calculate_2_body (initial_conditions,t,m1,m2, K1, K2):
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

def double_star_system (m1,m2,r1,r2, K1,K2, plot):
    m1 = m1/sun_mass
    m2 = m2/sun_mass
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
    res = integrate.odeint(calculate_2_body, initial_conditions, t, args=(m1,m2, K1, K2))
    r_1_pos = res[:,:3]
    r_2_pos = res[:,3:6]

    #---------------------------------PLOT------------------------------------------------------------
    #UNCOMMENT TO Plot from COM:
    r_com_final = ((m1*r_1_pos) +(m2*r_2_pos))/(m1+m2)
    r_1_pos = r_1_pos - r_com_final
    r_2_pos = r_2_pos - r_com_final

    if plot:
        fig=plt.figure(figsize=(10,10))
        ax=fig.add_subplot(111,projection="3d")

        ax.plot(r_1_pos[:,0],r_1_pos[:,1],r_1_pos[:,2],color="darkblue")
        ax.plot(r_2_pos[:,0],r_2_pos[:,1],r_2_pos[:,2],color="tab:red")
        ax.scatter(r_1_pos[-1,0],r_1_pos[-1,1],r_1_pos[-1,2],color="darkblue",marker="o",s=100,label="Star1")
        ax.scatter(r_2_pos[-1,0],r_2_pos[-1,1],r_2_pos[-1,2],color="tab:red",marker="o",s=100,label="Star2")
        plt.show()
    return r_1_pos, r_2_pos, r_com


def plot_total_system(system1_r1, system1_r2, system2_r1, system2_r2,total_system_r1, total_system_r2):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection="3d")

    # #PLOT SYSTEM 1
    ax.plot(system1_r1[:,0] - total_system_r1[-1,0],system1_r1[:,1] - total_system_r1[-1,1],system1_r1[:,2] - total_system_r1[-1,2],color="b")
    ax.plot(system1_r2[:,0] - total_system_r1[-1,0],system1_r2[:,1] - total_system_r1[-1,1],system1_r2[:,2] - total_system_r1[-1,2],color="g")
    ax.scatter(system1_r1[-1,0] - total_system_r1[-1,0],system1_r1[-1,1] - total_system_r1[-1,1],system1_r1[-1,2] - total_system_r1[-1,2],color="b",marker="o",s=100,label="Star1")
    ax.scatter(system1_r2[-1,0] - total_system_r1[-1,0],system1_r2[-1,1] - total_system_r1[-1,1],system1_r2[-1,2] - total_system_r1[-1,2],color="g",marker="o",s=100,label="Star2")
    # #PLOT SYSTEM 2
    # ax.plot(system2_r1[:,0],system2_r1[:,1],system2_r1[:,2],color="r")
    # ax.plot(system2_r2[:,0],system2_r2[:,1],system2_r2[:,2],color="c")
    # ax.scatter(system2_r1[-1,0],system2_r1[-1,1],system2_r1[-1,2],color="r",marker="X",s=100,label="Star3")
    # ax.scatter(system2_r2[-1,0],system2_r2[-1,1],system2_r2[-1,2],color="c",marker="X",s=100,label="Star4")
    ax.plot(system2_r1[:,0] + total_system_r1[-1,0],system2_r1[:,1] + total_system_r1[-1,1],system2_r1[:,2] + total_system_r1[-1,2],color="r")
    ax.plot(system2_r2[:,0] + total_system_r1[-1,0],system2_r2[:,1] + total_system_r1[-1,1],system2_r2[:,2] + total_system_r1[-1,2],color="c")
    ax.scatter(system2_r1[-1,0] + total_system_r1[-1,0],system2_r1[-1,1] + total_system_r1[-1,1],system2_r1[-1,2] + total_system_r1[-1,2],color="r",marker="o",s=100,label="Star3")
    ax.scatter(system2_r2[-1,0] + total_system_r1[-1,0],system2_r2[-1,1] + total_system_r1[-1,1],system2_r2[-1,2] + total_system_r1[-1,2],color="c",marker="o",s=100,label="Star4")

    # #PLOT TOTAL SYSTEM
    ax.plot(total_system_r1[:,0],total_system_r1[:,1],total_system_r1[:,2],color="y")
    ax.plot(total_system_r2[:,0],total_system_r2[:,1],total_system_r2[:,2],color="k")
    # ax.scatter(total_system_r1[-1,0],total_system_r1[-1,1],total_system_r1[-1,2],color="y",marker="X",s=100,label="Star_System_1")
    # ax.scatter(total_system_r2[-1,0],total_system_r2[-1,1],total_system_r2[-1,2],color="k",marker="X",s=100,label="Star_System_2")

    plt.show()
#------------------------------DOUBLE SYSTEM CALCULATIONS--------------------------------------------
m1 = 4.06e+30
m2 = 3.22e+30

m3 = 4.22e+30
m4 = 4.3e+30
system1_r1, system1_r2, system1_com = double_star_system (m1,m2,np.array([-0.5,0,0],dtype="float64"),np.array([0.5,0,0],dtype="float64"), K1_1, K2_1, False)
system2_r1, system2_r2, system2_com = double_star_system (m3,m4,np.array([1.5,0,0],dtype="float64"),np.array([2.5,0,0],dtype="float64"), K1_2, K2_2, True)

#------------------------------TOTAL SYSTEM CALCULATION--------------------------------------------
#
# #------------------------------MORE CONSTANTS-------------------------------------------------------
# orbital_period_3 = 79.91*365*24*3600*0.51 #in seconds
# distance_between_stars3 = 1.575e+12
# #LA.norm(system2_com - system1_com)
# K1_3 = (G*orbital_period_2*sun_mass)/(np.power(distance_between_stars2,2) * earth_sun_v)
# K2_3 = (earth_sun_v*orbital_period_2)/distance_between_stars2
#
# #--------------------------------------------------------------------------------------------
# total_system_r1, total_system_r2, total_system_com = double_star_system (m1+m2,m3+m4,system1_com,system2_com, K1_3, K2_3, False)
# #plot_total_system(system1_r1, system1_r2, system2_r1, system2_r2,total_system_r1, total_system_r2)
