import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.optimize

from ipydex import IPS


img_path = "/home/ck/mnt/XAI-DIA-gl/Julian/Dataset_Masterarbeit/atsds_large_ground_truth/train/00001/000000.png"
img_rgb = cv2.imread(img_path)[:, :, ::-1] // 255
img_2d = np.average(img_rgb, axis=2).astype(int)

N = 10

sum_value = np.sum(img_2d)
mask_size1 =  np.round(sum_value / N)
mask_sizes = np.array([mask_size1]*N)

# account for the remainder in the last mask
diff = sum_value - N*mask_size1
mask_sizes[-1] += diff


def smooth_round1(z):
    z_int  = np.floor(z)
    z_rest = z - z_int
    k = 4

    # only consider cases where z_rest != 0
    q = (z_rest != 0)
    sigma = np.zeros_like(z_rest)
    sigma[q] += 1/(1 + np.exp(-k*np.tan((z_rest[q]+0.5)*np.pi)))
    return z_int + sigma

def smooth_round(z):

    q = np.sin(2*np.pi*z)/(2*np.pi)
    return z - q


if 0:
    zz = np.linspace(0, 4, 500)
    plt.plot(zz, smooth_round(zz))
    plt.grid()
    plt.show()
    exit()


def find_product_pair(x):
    """
    Find a, b such that (x - a*b)**2 is minimal
    """

    def target(p):

        # smoothly approximate the round function:

        return (x - smooth_round(p[0])*smooth_round(p[1]))**2

    return scipy.optimize.minimize(target, (np.floor(x**.5),)*2)


res = find_product_pair(mask_size1)
IPS()

print(mask_sizes)




IPS()