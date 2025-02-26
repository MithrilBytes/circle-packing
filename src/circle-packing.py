import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class Circle:
    """Class representing a circle with position, radius, and color."""
    def __init__(self, x, y, r, color):
        self.x = x
        self.y = y
        self.r = r
        self.color = color  # Store the color of the pixel

def compute_image_features(image):
    """Computes local pixel density and edge strength for radius adjustment."""
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    density_map = 255 - blurred  # Inverted brightness to prioritize dark areas

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edge_map = cv2.magnitude(sobel_x, sobel_y)

    density_map = density_map / 255.0  # Normalize to 0-1
    edge_map = edge_map / np.max(edge_map)  # Normalize

    return density_map, edge_map

def generate_poisson_samples(image, num_samples, min_dist):
    """Generates Poisson-disc distributed points, prioritizing dark areas."""
    height, width = image.shape[:2]
    points = []
    grid_size = min_dist / np.sqrt(2)
    grid_width = max(1, int(width / grid_size))
    grid_height = max(1, int(height / grid_size))
    grid = -np.ones((grid_width, grid_height), dtype=int)

    def fits(x, y):
        """Checks if a point fits without violating min_dist."""
        gx, gy = min(grid_width - 1, max(0, int(x / grid_size))), min(grid_height - 1, max(0, int(y / grid_size)))
        for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
            for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                idx = grid[i, j]
                if idx != -1:
                    px, py = points[idx]
                    if np.linalg.norm((x - px, y - py)) < min_dist:
                        return False
        return True

    sorted_pixels = sorted([(x, y, image[y, x]) for y in range(height) for x in range(width)], key=lambda p: p[2])

    for x, y, brightness in sorted_pixels[:num_samples]:  
        if fits(x, y):
            points.append((x, y))
            gx, gy = min(grid_width - 1, max(0, int(x / grid_size))), min(grid_height - 1, max(0, int(y / grid_size)))
            grid[gx, gy] = len(points) - 1  

    return points

def process_image(image_path, num_circles=1000):
    """Loads an image, converts it to grayscale, and generates circle placements."""
    original_image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    if grayscale_image is None:
        print("Error: Could not load image.")
        return None, None

    height, width = grayscale_image.shape

    density_map, edge_map = compute_image_features(grayscale_image)

    points = generate_poisson_samples(grayscale_image, num_circles, min_dist=10)
    circles = []

    for x, y in points:
        brightness = grayscale_image[y, x]  
        density = density_map[y, x]  
        edge_strength = edge_map[y, x]  

        radius = max(3, int(20 * (1 - density) * (1 - edge_strength)))  

        color = original_image[y, x].tolist()  
        circles.append(Circle(x, y, radius, color))

    return original_image, circles

def get_coverage(circles, image_shape):
    """Calculates the percentage of the image covered by circles."""
    total_pixels = image_shape[0] * image_shape[1]  
    covered_pixels = sum(np.pi * (c.r ** 2) for c in circles)  
    return min(covered_pixels / total_pixels, 1.0)  

def fill_gaps_dynamically(circles, image, max_passes=10, min_radius=10):
    """Dynamically adds circles until the image reaches 95% coverage or max passes is hit."""
    height, width = image.shape[:2]
    pass_num = 1

    while get_coverage(circles, (height, width)) < 0.95 and pass_num <= max_passes:
        print(f"Pass {pass_num}: Filling gaps with min radius {min_radius}")
        additional_circles = []

        for _ in range(len(circles) // 2):  
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            brightness = np.mean(image[y, x])  
            radius = max(min_radius, int((255 - brightness) / 25))  
            color = image[y, x].tolist()  

            overlap = any(np.linalg.norm((x - c.x, y - c.y)) < radius + c.r for c in circles)
            if not overlap:
                additional_circles.append(Circle(x, y, radius, color))

        circles.extend(additional_circles)
        min_radius = max(2, min_radius // 1.5)  
        pass_num += 1

def draw_circles(image, circles, output_path):
    """Draws packed circles colored based on image pixels."""
    height, width = image.shape[:2]
    packed_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    for circle in circles:
        radius = int(circle.r)  
        cv2.circle(packed_image, (circle.x, circle.y), radius, circle.color, -1)  

    plt.imshow(cv2.cvtColor(packed_image, cv2.COLOR_BGR2RGB))  
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

def main():
    image_path = input("Enter the full path to the image file: ").strip()
    num_circles = int(input("Enter the number of initial circles (e.g., 1000): "))

    image, circles = process_image(image_path, num_circles)
    if image is None or circles is None:
        print("Error in processing image. Exiting...")
        return

    fill_gaps_dynamically(circles, image, max_passes=10, min_radius=15)  

    output_path = "output/circle_packed.png"
    draw_circles(image, circles, output_path)

    print(f"Circle packed image saved to {output_path}")

if __name__ == "__main__":
    main()