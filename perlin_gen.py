import noise
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
import random


def generate_perlin_noise_csv(
    n, filename, scale=10, octaves=1, persistence=0.5, lacunarity=2.0
):
    # create empty n x n grid to hold noise values
    noise_grid = np.zeros((n, n))

    # set random seed for noise generation
    random_z = random.randint(0, 10000)

    # generate perlin noise for each cell in the grid
    for i in range(n):
        for j in range(n):
            # scale coordinates to generate more detailed or smoother noise
            x = i / scale
            y = j / scale
            z = random_z
            noise_value = noise.pnoise3(
                x,
                y,
                z,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=n,
                repeaty=n,
                base=1,
            )
            noise_grid[i][j] = noise_value

    flat_noise = noise_grid.flatten()

    # get sorted indices to preserve original positions
    sorted_indices = np.argsort(flat_noise)

    # map noise to a distribution
    cdf_values = np.linspace(0, 1, len(flat_noise), endpoint=False)
    mapped_values = norm(loc=0, scale=1).ppf(cdf_values)
    mapped_values = np.clip(mapped_values, -3, 3)  # set by experimentation

    # create new array to hold the mapped values in the original positions
    remapped_noise = np.zeros_like(flat_noise)

    # assign mapped values back to their original positions
    remapped_noise[sorted_indices] = mapped_values

    # reshape to original grid shape
    noise_grid = remapped_noise.reshape(noise_grid.shape)

    # rescale to [0, 1]
    min_val = noise_grid.min()
    max_val = noise_grid.max()
    print(min_val, max_val)
    noise_grid = (noise_grid - min_val) / (max_val - min_val)

    # write noise grid to a CSV file
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in noise_grid:
            writer.writerow(row)

    print(f"Perlin noise CSV generated and saved as {filename}")

    # visualize noise grid and distribution of values
    plt.imshow(noise_grid, cmap="gray", interpolation="nearest")
    plt.colorbar(label="Noise value")
    plt.title(f"Perlin Noise (n={n}, scale={scale}, octaves={octaves})")
    plt.show()

    plt.hist(noise_grid.flatten(), bins=50, color="blue", alpha=0.7)
    plt.title("Noise distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


# Example usage:
generate_perlin_noise_csv(51, "perlin-noise.csv", scale=20, octaves=2)
