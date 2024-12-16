import numpy as np
import cv2
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from voronoi import plot_polygon_like_obj

from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()


# Create a sample image (300x300 with 70% black and 30% white)
image = np.zeros((300, 300), dtype=np.uint8)
image[50:200, 100:200] = 255  # Create a white rectangle

cv2.circle(image, (190, 100 ), 50, color=255, thickness=-1)


# img_path = "/home/ck/mnt/XAI-DIA-gl/Julian/Dataset_Masterarbeit/atsds_large_ground_truth/train/00001/000000.png"
# img_rgb = cv2.imread(img_path)[:, :, ::-1] // 255
# img_2d = np.average(img_rgb, axis=2).astype(int)



# Find contours
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Approximate the contour
epsilon = 0.001 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)

# Convert to Shapely polygon
pg = Polygon(approx.reshape(-1, 2))

edges = np.array(pg.edges(mode="tuples")).astype(int)

for e in edges:
    cv2.line(image, *e, color=100)
cv2.imwrite("tmp.png", image)


# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

plot_polygon_like_obj(None, pg)
plt.show()
# plt.savefig("tmp.png")

exit()
x, y = polygon.exterior.xy
ax2.plot(x, y, color='red', linewidth=2)
ax2.set_title('Approximated Polygon')
ax2.set_xlim(0, 300)
ax2.set_ylim(300, 0)  # Invert y-axis to match image coordinates
ax2.axis('off')

plt.tight_layout()
plt.show()
