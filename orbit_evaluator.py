import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import permutations


def calc_magnitude(vec):
    return np.abs(np.power(vec[0],2) + np.power(vec[1],2) + np.power(vec[2],2))

def calc_residual_1(linepts, orbit_points):
    #TODO: Figure out how to measure the error between lobf and elipse
    direction_v = np.array([linepts[0][0] - linepts[1][0], linepts[0][1] - linepts[1][1], linepts[0][2] - linepts[1][2]])

    residual_sum = 0
    residual_mag_sum = 0
    #tryna multiply the direction vector by all the orbit X cords but shits not working
    for point in orbit_points:
        lobf_vector = linepts[2] + (direction_v * point[0])
        residual = np.abs(point - lobf_vector)

        residual_mag_sum += calc_magnitude(residual)
        residual_sum+=residual

    print (calc_magnitude(residual_sum))
    print (residual_mag_sum)
    #print (np.mgrid[-100:100:3j][:, np.newaxis])

def calc_residual_2(orbit_points, avg_point_cloud, direction):
    residual_sum = 0
    residual_mag_sum = 0
    #tryna multiply the direction vector by all the orbit X cords but shits not working
    for point in orbit_points:
        lobf_vector = (direction * point[0]) + avg_point_cloud
        residual = np.abs(point - lobf_vector)
        residual_mag_sum += calc_magnitude(residual)
        residual_sum+=residual

    print (calc_magnitude(residual_sum))
    print (residual_mag_sum)
    #print (np.mgrid[-100:100:3j][:, np.newaxis])

def SLR (points):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    X = np.array([y,z]).T
    res = LinearRegression().fit(X,x)
    score = res.score(X,x)
    return score

def SLR_2(points):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    X = np.array([x,y,z])

    #https://math.stackexchange.com/questions/1611308/best-fit-line-with-3d-points/1612462#1612462
    #https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d

    #calculate center of cloud (mean of all points)
    avg_point_cloud = np.array([np.sum(x), np.sum(y), np.sum(z)])/X.shape[0]
    #center all the data points around the mean
    mean_centered_data = [val - avg_point_cloud for val in X.T]
    #factor matrix
    uu,dd,vv = np.linalg.svd(mean_centered_data)

    #vv[0] contains the line direction
    calc_residual_2(points,avg_point_cloud, vv[0])

    #calc line of best fit
    #TODO: FIND SPREAD OF DATA INSTEAD OF HARDCORDING [-60:60]
    linepts = vv[0] * np.mgrid[-100:100:3j][:, np.newaxis]
    #shift line back my mean
    linepts += avg_point_cloud
    linepts = linepts.T
    return linepts

def velocity_finder_bf ():
    #get all possible velocoites
    #assumes -2 <= v <= 2 with increments of 0.1
    # possible_velocities = [0]
    # for i in range (1,):
    #     possible_velocities.append(i/10)
    # possible_velocities.extend(list(np.array(possible_velocities.copy()) * -1))
    possible_velocities = [0,0.03,0.05,0.1,0.02,0.01, -0.03,-0.05,-0.1,-0.02,-0.01,]
    #[0,0.25,0.5,1,1.25,1.5,2.0,-0.25,-0.5,-1,-1.25,-1.5,-2.0]
    perm = permutations(possible_velocities,6)
    perm = list(perm)
    return perm
