import os
from PIL import Image

# Paths
input_path = r"C:\Users\lizhi\Downloads\liif-main\liif\data\single\valid\input.jpg"
output_dir = r"C:\Users\lizhi\Downloads\liif-main\liif\data"

# Load image
img = Image.open(input_path)
W, H = img.size

# Grid parameters
rows, cols = 4, 4
tile_w = W // cols
tile_h = H // rows

# Prepare folders
br_folder = os.path.join(output_dir, "bottom_right_quadrant")
other_folder = os.path.join(output_dir, "other_quadrants")
os.makedirs(br_folder, exist_ok=True)
os.makedirs(other_folder, exist_ok=True)

# Split and save
tiles_br = []
tiles_other = []

for i in range(rows):
    for j in range(cols):
        # Crop tile
        left = j * tile_w
        upper = i * tile_h
        right = (j + 1) * tile_w
        lower = (i + 1) * tile_h
        tile = img.crop((left, upper, right, lower))

        # Determine quadrant: bottom-right quadrant is i//2 == 1 and j//2 == 1
        if i // 2 == 1 and j // 2 == 1:
            tiles_br.append(((i, j), tile))
        else:
            tiles_other.append(((i, j), tile))

# Sort by (row, col) and save
for (i, j), tile in sorted(tiles_br, key=lambda x: (x[0][0], x[0][1])):
    tile.save(os.path.join(br_folder, f"{i}_{j}.png"))

for (i, j), tile in sorted(tiles_other, key=lambda x: (x[0][0], x[0][1])):
    tile.save(os.path.join(other_folder, f"{i}_{j}.png"))

print("Tiles saved:")
print(f" - Bottom-right quadrant (4 tiles) in: {br_folder}")
print(f" - Other quadrants (12 tiles) in:   {other_folder}")
