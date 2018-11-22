import sol3 as sol

im = sol.imread("gray_orig.png")
norm_filter, filter_base = sol.create_blur_filter(15)
norm_filter_base = filter_base / sol.np.sum(filter_base)
im1 = sol.scipy.ndimage.filters.convolve(im, norm_filter, mode='mirror')
im2 = sol.scipy.ndimage.filters.convolve(im, norm_filter_base, mode='mirror')
im2 = sol.scipy.ndimage.filters.convolve(im2, norm_filter_base.T, mode='mirror')
print(im1)
print(im2)
print(sol.np.allclose(im1, im2))

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(im1, cmap=plt.cm.gray)
plt.figure()
plt.imshow(im2, cmap=plt.cm.gray)
plt.show()
