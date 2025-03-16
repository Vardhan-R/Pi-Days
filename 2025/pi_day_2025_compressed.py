import matplotlib.pyplot as plt, numpy as np
class Organism:
    def __init__(self, n_steps: int | None = None, angles: np.ndarray | None = None) -> None: self.angles = np.atan2(rng.uniform(-1, 1, n_steps), rng.uniform(-1, 1, n_steps)) if angles is None else angles.copy()
    def live(self, step_len: float | int) -> float:
        self.points = np.cumsum(np.vstack((np.zeros((1, 2)), np.column_stack((step_len * np.cos(self.angles), step_len * np.sin(self.angles))))), axis=0)
        return abs(sum(curr_pt[0] * prev_pt[1] - curr_pt[1] * prev_pt[0] for prev_pt, curr_pt in zip(self.points[:-1], self.points[1:])) / 2)
rng, n_steps, length, n_orgs, n_gens = np.random.default_rng(), 100, 1, 100, 901
orgs = [Organism(n_steps) for _ in range(n_orgs)]
for gen in range(n_gens):
    areas, best_ind = (lambda arr: (arr, np.argmax(arr)))([org.live(length / n_steps) for org in orgs])
    print(f"Gen {gen}: lowest pi = {length ** 2 / (2 * areas[best_ind])}")
    if gen % 100 == 0: plt.plot(orgs[best_ind].points[:, 0], orgs[best_ind].points[:, 1], label=f"Gen {gen}")
    orgs = [Organism(angles=orgs[best_ind].angles + (rng.uniform(0, 1, orgs[best_ind].angles.size) < 0.01) * rng.normal(0, 0.1, orgs[best_ind].angles.size)) for _ in range(n_orgs)]
plt.legend()
plt.axis("equal")
plt.title("Paths")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()
best_angles = orgs[best_ind].angles % (2 * np.pi) - np.pi
print(np.max(best_angles) - np.min(best_angles))
plt.plot(best_angles)
plt.ylim(-np.pi, np.pi)
plt.yticks(np.linspace(-np.pi, np.pi, 13))
plt.title("Angle vs. Time")
plt.xlabel("Time")
plt.ylabel("Angle")
plt.show()
