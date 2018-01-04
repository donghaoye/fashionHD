import util.image as image

img = image.imread('demo.jpg')

p_src = [(10,10), (10,20), (20,10), (20,20)]
p_tar = [(20,20), (20,40), (40,20), (40,40)]

img_out = image.align_image(img, p_src, p_tar, sz_tar = (1024, 1024))

image.imwrite(img_out, 'aligned.jpg')
