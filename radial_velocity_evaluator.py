import numpy as np

def radial_velocity(e,a,i,P, theta, omega):
    #http://www.relativitycalculator.com/pdfs/RV_Derivation.pdf
    #p.25
    c1 = (2*3.14*a*np.sin(i))/(P*np.sqrt(1-np.power(e,2)))
    c2 = np.cos(theta+omega) + (e*np.cos(omega))
    return c1*c2

def lists2dict(list1, list2):
    dictionary = {}
    for val1, val2 in list(zip(list1, list2)):
        dictionary[val1] = val2
    return dictionary

def rad2degrees(rad):
    return (180*rad)/3.14

def get_semi_major_acess (x,y,z):
    x2yz = lists2dict(x,np.column_stack((y,z)))
    y2xz = lists2dict(y,np.column_stack((x,z)))

    max_x = np.max(x)
    max_x_cords = [max_x]
    max_x_cords.extend(x2yz.get(max_x))
    min_x = np.min(x)
    min_x_cords = [min_x]
    min_x_cords.extend(x2yz.get(min_x))

    max_y = np.max(y)
    max_y_cords = [y2xz.get(max_y)[0], max_y, y2xz.get(max_y)[1]]
    min_y = np.min(y)
    min_y_cords = [y2xz.get(min_y)[0], max_y, y2xz.get(min_y)[1]]

    a1_dist = np.linalg.norm(np.array(max_x_cords) - np.array(min_x_cords))
    a2_dist = np.linalg.norm(np.array(max_y_cords) - np.array(min_y_cords))
    #get largest axis
    if a1_dist > a2_dist:
        return a1_dist
    else:
        return a2_dist

def get_eccentricity_vector(v,r,m):
    #https://en.wikipedia.org/wiki/Eccentricity_vector
    u = 6.67408e-11*m*2e30
    p1 = ((np.power(np.linalg.norm(v),2)/u) - (1/np.linalg.norm(r)))*r
    p2 = (np.dot(r,v)/u)*np.array(v)

    p1 = np.array(p1)
    p2 = np.array(p2)
    return p1 - p2

def get_true_anamoly(e_v,v,r):
    return rad2degrees(np.arccos(np.dot(e_v,r)/np.linalg.norm(e_v-r)))

def get_argument_of_periastron(h,e_v):
    #https://en.wikipedia.org/wiki/Argument_of_periapsis
    #vector pointing towards ascending node
    n = h*np.array([1,0,0])
    return rad2degrees(np.arccos(np.dot(n,e_v)/(np.linalg.norm(n)*np.linalg.norm(e_v))))

def get_inclination(r,v,m):
    #orbital momentum vector
    h = (np.power(np.linalg.norm(r),2)*m) * ((r*v)/np.power(np.linalg.norm(r),2))

    return rad2degrees(np.arccos(h[2]/np.linalg.norm(h))), h

def evaluate (x,y,z,e,P,v_all,r,m):
    #devide major access by 2 to get semi major access
    a =  (get_semi_major_acess (x,y,z))/2
    for v in v_all:
        #get eccentricity vector
        e_v = get_eccentricity_vector(v,r,m)
        #find true anomoly
        theta = get_true_anamoly(e_v,v,r)
        #find inclination
        i, h = get_inclination(r,m,v)
        #find argument of periaston
        omega = get_argument_of_periastron(h, e_v)
        r_v = radial_velocity(e,a,i,P, theta, omega)
