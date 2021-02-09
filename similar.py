import image
from PIL import Image
import numpy
import os

class ImageHash(object):
	"""
	Hash encapsulation. Can be used for dictionary keys and comparisons.
	"""
	def __init__(self, binary_array):
		self.hash = binary_array

	def __str__(self):
		return _binary_array_to_hex(self.hash.flatten())

	def __repr__(self):
		return repr(self.hash)

	def __sub__(self, other):
		if other is None:
			raise TypeError('Other hash must not be None.')
		if self.hash.size != other.hash.size:
			raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
		return numpy.count_nonzero(self.hash.flatten() != other.hash.flatten())

	def __eq__(self, other):
		if other is None:
			return False
		return numpy.array_equal(self.hash.flatten(), other.hash.flatten())

	def __ne__(self, other):
		if other is None:
			return False
		return not numpy.array_equal(self.hash.flatten(), other.hash.flatten())

	def __hash__(self):
		# this returns a 8 bit integer, intentionally shortening the information
		return sum([2**(i % 8) for i, v in enumerate(self.hash.flatten()) if v])


def average_hash(image, hash_size=9):
	"""
	Average Hash computation

	Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

	Step by step explanation: https://www.safaribooksonline.com/blog/2013/11/26/image-hashing-with-python/

	@image must be a PIL instance.
	"""
	if hash_size < 0:
		raise ValueError("Hash size must be positive")

	# reduce size and complexity, then covert to grayscale
	image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)

	# find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
	pixels = numpy.asarray(image)
	avg = pixels.mean()

	# create string of bits
	diff = pixels > avg
	# make a hash
	return ImageHash(diff)


def is_image(filename):
    f = filename.lower()
    return f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".bmp") or f.endswith(".gif")

def find_similar_images(path):
    count = 0
    image_filenames = [os.path.join(path, u_path) for u_path in os.listdir(path) if is_image(u_path)]
    images = {}
    for img in sorted(image_filenames):
        hash = average_hash(Image.open(img))
        images[hash] = images.get(hash, []) + [img]
    for k, img_list in images.items():
        if len(img_list) > 1:
            print(" ".join(img_list))
            print("\n")
            count += 1
    return count
path = input("enter path\n")
count = find_similar_images(path)
print("n.of similar image sets are " + str(count))
