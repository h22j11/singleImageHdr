import cv2
image = cv2.imread(r'training_data\Test\Bracketed_images\HDR001_1800x600_80_10_0_gamma+1.4\e.png')
new_width = 256
new_height = 256
resized_image = cv2.resize(image, (new_width, new_height))
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.imwrite('raw.jpg', resized_image)
cv2.destroyAllWindows()