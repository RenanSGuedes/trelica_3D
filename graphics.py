import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(facecolor=plt.cm.Blues(0.2))
ax = fig.add_subplot(111, projection="3d")
fig.suptitle(
    f"3D-vector (2,3,4)",
    fontsize=18,
    fontweight="bold",
)

xs, ys, zs = zip([2, 2, 2], [0, 0, 0])
ax.plot(xs, ys, zs, color="blue")

xs, ys, zs = zip([6, 1, 2], [1, 7, 1])
ax.plot(xs, ys, zs, color="#7159c1")

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

plt.show()
