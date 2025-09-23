from google.colab import files 
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

print("masukan file foto")
uploaded=files.upload()

# baca gambar
filename =list(uploaded.keys())[0]
image = cv2.imread(filename)
img_rgb=cv2.cvtcolor(image,cv2.COLOR_BGR2RGB)

# Filter Blur
kernel=np.ones((5,5),np.float32)/25
blur=cv2.filter2D(img_rgb, -1, kernel)

# Gausian Blur
gaussian =cv2.GaussianBlur(img_rgb,(5,5),0)

# Median Filter
median=cv2.medianBlur(img_rgb,5)

# Sharpening
sharpe_kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpen=cv2.filter2D(img-rgb,-1,sharpe_kernel)

# Edge 
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sobelx= cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobely= cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
sobel_combine=cv2.magnitude(sobelx,sobely)


# tampilkan
plt.figure(figsize=(14,8))

plt.subplot(2,3,1);plt.imshow(img_rgb);plt.title("Ini Ori");plt.axis("off")
plt.subplot(2,3,2);plt.imshow(blur);plt.title("Ini Blur");plt.axis("off")
plt.subplot(2,3,3);plt.imshow(gaussian);plt.title("Ini Gaussian");plt.axis("off")
plt.subplot(2,3,4);plt.imshow(median);plt.title("Ini Median");plt.axis("off")
plt.subplot(2,3,5);plt.imshow(sharpen);plt.title("Ini Sharpen");plt.axis("off")
plt.subplot(2,3,6);plt.imshow(sobel_combine,cmap='gray');plt.title("Ini Sobel");plt.axis("off")

plt.tight_layout()
plt.show()