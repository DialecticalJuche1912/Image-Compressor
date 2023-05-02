import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image
import requests  # Added requests import
from io import BytesIO  # Added BytesIO import

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception if there's an error
    return Image.open(BytesIO(response.content))

url = 'https://static.wikia.nocookie.net/supermarioglitchy4/images/f/f3/Big_chungus.png/revision/latest?cb=20200511041102'
img = download_image(url)

plt.axis('off')
plt.imshow(img)
plt.show()  # Added plt.show() to display the image

img_hold = img.convert('LA')
img_show = plt.imshow(img_hold)

img2Dmat = np.array(list(img_hold.getdata(band=0)), float)
img2Dmat.shape = (img_hold.size[1], img_hold.size[0])
img2Dmat_1 = np.matrix(img2Dmat)
plt.imshow(img2Dmat_1, cmap='gray')

U, Σ, VT = np.linalg.svd(img2Dmat)
reconstimg = np.matrix(U[:, :1]) * np.diag(Σ[:1]) * np.matrix(VT[:1, :])
plt.imshow(reconstimg, cmap='gray')

# Approximation of matrix at different ranks 
U, Σ, VT = np.linalg.svd(img2Dmat_1,full_matrices=False) 

#full_matrices=false means returning the more computationally efficient matrix () rather than the huge and inefficient m by m 
i =0 
for i in [2, 4, 8, 16, 32, 64]: 
    imgApprox = np.matrix(U[:, :i]) * np.diag(Σ[:i]) * np.matrix(VT[:i, :]) # take the frist i columns of times the first 
    plt.figure(i+1)
    plt.imshow(imgApprox, cmap='gray') # equivalent to Brunton's img = plt.imshow(Xapprox)
    plt.title("i = %s rank approx." % i)
    plt.axis('off') # turn off axis 
    plt.show()
    # Essentially, this only stores the first 2,4,8... columns of U and V in the first 100 diagonal elements of Σ, which will be sth like an 8 times comopression of original matrix X

img2Dmat.shape 
original_size = 1236*833
original_size
revised_size = 64*1236 + 64 + 64*833
revised_size
revised_size/original_size

# diagnoal elements of Σ, usually in a logj vs j graph 
plt.figure(1)
plt.semilogy(np.diag(Σ))
plt.title('Singular Values')
plt.show()
# note that the first few modes are capturing mot of the energy, i.e. the singular values between 1-200 is much much greater than the singular values from 600 to 800
# this means we can get away from throwing away only the first, say 200 and the remaining a sort of irrelevant 

# the cumulative sum of all Σj from j=1:i divided by sum of Σj from j=1:m 
# tells you how much of the matrix is captured by  
plt.figure(2)
plt.plot(np.cumsum(np.diag(Σ))/np.sum(np.diag(Σ)))
plt.title('Cumulative sum of the singular values')
plt.show()
# this graph shows that by keeping just the first vector would capture around 30% of the entire energy 
# dont need to keep all the vectors in a high res image, thus we can compress it using the SVD algorithm

