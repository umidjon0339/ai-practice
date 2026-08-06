import cv2

# ===============================
# Load Image
# ===============================

image = cv2.imread("images/sample.jpg")

if image is None:
    print("Image not found!")
    exit()

# ===============================
# Original Information
# ===============================

print("Height :", image.shape[0])
print("Width  :", image.shape[1])
print("Channels :", image.shape[2])

# ===============================
# Grayscale
# ===============================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ===============================
# Blur
# ===============================

blur = cv2.GaussianBlur(image, (9, 9), 0)

# ===============================
# Edge Detection
# ===============================

edges = cv2.Canny(gray, 100, 200)

# ===============================
# Resize
# ===============================

resized = cv2.resize(image, (500, 350))

# ===============================
# Rotate
# ===============================

height, width = image.shape[:2]

center = (width // 2, height // 2)

matrix = cv2.getRotationMatrix2D(center, 45, 1.0)

rotated = cv2.warpAffine(image, matrix, (width, height))

# ===============================
# Flip
# ===============================

flip_horizontal = cv2.flip(image, 1)
flip_vertical = cv2.flip(image, 0)

# ===============================
# Save Images
# ===============================

cv2.imwrite("output/grayscale.jpg", gray)
cv2.imwrite("output/blur.jpg", blur)
cv2.imwrite("output/edges.jpg", edges)
cv2.imwrite("output/resized.jpg", resized)
cv2.imwrite("output/rotated.jpg", rotated)
cv2.imwrite("output/flip_horizontal.jpg", flip_horizontal)
cv2.imwrite("output/flip_vertical.jpg", flip_vertical)

# ===============================
# Show Windows
# ===============================

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Edges", edges)
cv2.imshow("Resized", resized)
cv2.imshow("Rotated", rotated)
cv2.imshow("Horizontal Flip", flip_horizontal)
cv2.imshow("Vertical Flip", flip_vertical)

print("\nPress any key to exit...")

cv2.waitKey(0)
cv2.destroyAllWindows()