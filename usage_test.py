
from area_coverage import AreaPolygon, plot_coordinates, plot_path
import matplotlib.pyplot as plt

# Define polygon boundary (pentagon example)
exterior = [(0, 0), (40, 40), (0, 80), (-40, 40), (-90, 30)]
interior = [[(0,50),(-25,25),(13,40),(14,45)]]  # Example hole (interior polygon)
# Create area polygon for coverage planning
polygon = AreaPolygon(
    coordinates=exterior,
    initial_pos=(-5, 10),  # Starting UAV position
    interior=interior,                   # No holes in the area
    path_spacing=2,                # Path spacing (swath width)
    fixed_angle=None               # Path angle in degrees (None = auto-compute)
)

# Generate coverage path
path = polygon.generate_coverage_path(custom_origin=(0.0, 0.0))

# Visualize
fig, ax = plt.subplots()
plt.plot(*polygon.rP.exterior.xy, 'b-', label='Coverage Area')
plot_path(ax, path, color='red')
plt.plot(*polygon.P.exterior.xy, 'g--', label='Original')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()