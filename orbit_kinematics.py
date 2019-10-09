import numpy as np


def calc_center_of_mass (m1,m2):
    return (((-0.5 * m1) + (0.5 * m2))/ (m1+m2))

def euclid_distance(cords1, cords2):
    x1,y1 = cords1
    x2,y2 = cords2
    return np.sqrt(np.power(x2-x1,2) + np.power(y2-y1,2))

def calc_init_velocity(m, r, eps, s_cords, com_cords):
    G = 6.7 * (1/np.power(10,11))
    R = euclid_distance(s_cords, com_cords)
    return np.sqrt((G*m*R*(1+eps))/np.power(r,2))

def calc_acceleration(s1,s2, m,r):
    G = 6.7 * (1/np.power(10,11))
    #TODO: idk where this scalar comes in but lets see if it works
    u_scalar = (s2-s1)/np.abs(s2-s1)
    a = (G*m)/np.power(r,2)
    return a*u_scalar

def update_star_pos(p1_last_s, p2_last_s, m1, m2, m1_r, m2_r, last_v,t):
    #TODO: Check to make sure if the U scalar is necessary at this step
    R = euclid_distance(p1_last_s, p2_last_s)
    a = calc_acceleration(p1_last_s,p2_last_s,m2,R)
    current_v = a*t + last_v
    current_pos = (a*np.power(t,2)) + (current_v * t) + p1_last_s
    return current_pos, current_v

m1 = 1000
m1_r = 100

m2 = 500
m2_r = 50

def main(m1,m1_r, m2, m2_r):
    max_its = 10
    max_frames = 100
    frame = 0

    com_x = calc_center_of_mass (m1,m2)
    #TODO: idk what to do with Y values
    com_cords = (com_x, 0)

    #calculates initial star positions
    S_x1 = -0.5 - com_x
    s1_cords = (S_x1, 0)
    S_x2 = 0.5 - com_x
    s2_cords = (S_x2, 0)

    #calculate intital star velocities
    p1_v_init = calc_init_velocity(m2, m1_r, 0.8, s1_cords, com_cords)
    #stars must have opposite velocities so multiply by -1
    p2_v_init = calc_init_velocity(m1, m2_r, 0.8, s2_cords, com_cords) * -1

    p1_last_s = S_x1
    p1_last_v = p1_v_init

    p2_last_s = S_x2
    p2_last_v = p2_v_init

    while frame < max_frames:
        i = 0
        while i<max_its:
            #update star positions
            p1_n_pos, p1_n_v = update_star_pos(p1_last_s, p2_last_s, m1, m2, m1_r, m2_r, p1_last_v,1)
            p2_n_pos, p2_n_v = update_star_pos(p2_last_s, p1_last_s, m2, m1, m2_r, m1_r, p2_last_v,1)
            print (p1_n_pos)
            print (p1_n_v)
            print ("\n")
            print (p2_n_pos)
            print (p2_n_v)
            print ("-------------------------------------")


            p1_last_s = p1_n_pos
            p1_last_v = p1_n_v

            p2_last_s = p2_n_pos
            p2_last_v = p2_n_v

            i+=1
        frame+=1

main(m1,m1_r, m2, m2_r)

# #calculate initial acceleration
# p1_a_init = calc_acceleration(S_x1,S_x2, m2,m1_r)
# p2_a_init = calc_acceleration(S_x2,S_x1, m1,m2_r)
