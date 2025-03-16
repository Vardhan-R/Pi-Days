import matplotlib.pyplot as plt
import numpy as np

class Organism:
    def __init__(self, n_steps: int | None = None, angles: np.ndarray | None = None) -> None:
        if angles is None:
            x, y = rng.uniform(-1, 1, (2, n_steps))
            self.angles = np.atan2(x, y)
            return
        self.angles = angles.copy()

    def move(self, step_len: float | int) -> None:
        steps = np.column_stack((step_len * np.cos(self.angles), step_len * np.sin(self.angles)))
        self.points = np.cumsum(np.vstack((np.zeros((1, 2)), steps)), axis=0)

        # x_steps = step_len * np.cos(self.angles)
        # y_steps = step_len * np.sin(self.angles)
        # self.points = np.zeros((self.angles.size + 1, 2))
        # for i, (x, y) in enumerate(zip(x_steps, y_steps)):
        #     self.points[i:, 0] += x
        #     self.points[i:, 1] += y

        # for cos_ang, sin_ang in zip(cos_angs, sin_angs):
        #     new_pt = Point2D(step_len * cos_ang, step_len * sin_ang)
        #     self.points.append(self.points[-1] + new_pt)

    def calcArea(self) -> float:
        # vels = [Point2D(0, 0)]
        # for prev_pt, curr_pt in zip(self.points[:-1], self.points[1:]):
        #     vels.append(curr_pt - prev_pt)
        # return abs(sum(pt.x * vel.y - pt.y * vel.x for pt, vel in zip(self.points, vels)) / 2)
        return abs(sum(curr_pt[0] * prev_pt[1] - curr_pt[1] * prev_pt[0] for prev_pt, curr_pt in zip(self.points[:-1], self.points[1:])) / 2)

# class Point2D:
#     def __init__(self, x: float | int, y: float | int) -> None:
#         self.x = x
#         self.y = y

#     def __add__(self, other: "Point2D") -> "Point2D":
#         return Point2D(self.x + other.x, self.y + other.y)

#     def __sub__(self, other: "Point2D") -> "Point2D":
#         return Point2D(self.x - other.x, self.y - other.y)

def asexualReproduction(org: Organism, mutation_rate: float | int = 0.01, std_dev: float | int = 0.1) -> Organism:
    new_org = Organism(angles=org.angles)
    new_org.angles += (rng.uniform(0, 1, new_org.angles.size) < mutation_rate) * rng.normal(0, std_dev, new_org.angles.size)
    return new_org

rng = np.random.default_rng()

n_steps = 100
length = 1
step_len = length / n_steps
n_orgs = 100
n_gens = 901

orgs = [Organism(n_steps) for _ in range(n_orgs)]

areas = np.zeros(n_orgs)
for gen in range(n_gens):
    for i, org in enumerate(orgs):
        org.move(step_len)
        areas[i] = org.calcArea()

    best_ind = np.argmax(areas)
    print(f"Gen {gen}: lowest pi = {length ** 2 / (2 * areas[best_ind])}")

    best_org = orgs[best_ind]
    orgs = [asexualReproduction(best_org) for _ in range(n_orgs)]

    if gen % 100 == 0:
        plt.plot(best_org.points[:, 0], best_org.points[:, 1], label=f"Gen {gen}")

plt.legend()
plt.axis("equal")
plt.title("Paths")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()

best_angles = best_org.angles
best_angles %= 2 * np.pi
best_angles -= np.pi
print(np.max(best_angles) - np.min(best_angles))

plt.plot(best_angles)
plt.ylim(-np.pi, np.pi)
plt.yticks(np.linspace(-np.pi, np.pi, 13))
plt.title("Angle vs. Time")
plt.xlabel("Time")
plt.ylabel("Angle")
plt.show()
