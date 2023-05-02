import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def compress_image(img, k):
    img_array = np.array(img)
    red, green, blue = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    U_r, Σ_r, VT_r = np.linalg.svd(red, full_matrices=False)
    U_g, Σ_g, VT_g = np.linalg.svd(green, full_matrices=False)
    U_b, Σ_b, VT_b = np.linalg.svd(blue, full_matrices=False)

    red_approx = (U_r[:, :k] @ np.diag(Σ_r[:k]) @ VT_r[:k, :]).clip(0, 255).astype(np.uint8)
    green_approx = (U_g[:, :k] @ np.diag(Σ_g[:k]) @ VT_g[:k, :]).clip(0, 255).astype(np.uint8)
    blue_approx = (U_b[:, :k] @ np.diag(Σ_b[:k]) @ VT_b[:k, :]).clip(0, 255).astype(np.uint8)

    img_approx = np.stack([red_approx, green_approx, blue_approx], axis=-1)
    return Image.fromarray(img_approx)

url = 'https://static.wikia.nocookie.net/supermarioglitchy4/images/f/f3/Big_chungus.png/revision/latest?cb=20200511041102'
img = download_image(url)

plt.figure(0)
plt.axis('off')
plt.imshow(img)

for i in [64, 32, 16, 8, 4, 2]:
    img_approx = compress_image(img, i)
    plt.figure(i+1)
    plt.imshow(img_approx)
    plt.title("i = %s rank approx." % i)
    plt.axis('off')

plt.show()

plt.figure(1)
plt.semilogy(np.diag(Σ_r), label='Red')
plt.semilogy(np.diag(Σ_g), label='Green')
plt.semilogy(np.diag(Σ_b), label='Blue')
plt.title('Singular Values')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(Σ_r))/np.sum(np.diag(Σ_r)), label='Red')
plt.plot(np.cumsum(np.diag(Σ_g))/np.sum(np.diag(Σ_g)), label='Green')
plt.plot(np.cumsum(np.diag(Σ_b))/np.sum(np.diag(Σ_b)), label='Blue')
plt.title('Cumulative sum of the singular values')
plt.legend()
plt.show()
