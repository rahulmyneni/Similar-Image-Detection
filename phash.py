import image
from imagehash import *

def find_similar_images(userpath, hashfunc):
	import os
	def is_image(filename):
		f = filename.lower()
		return f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".bmp") or f.endswith(".gif")
	count = 0
	image_filenames = [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
	images = {}
	for img in sorted(image_filenames):
		hash = hashfunc(Image.open(img))
		images[hash] = images.get(hash, []) + [img]

	for k, img_list in images.items():
		if len(img_list) > 1:
			sys.stderr.write(" ".join(img_list))
			sys.stderr.write("\n")
			count += 1
	return count


if __name__ == '__main__':
	import sys, os
	def usage():
		sys.stderr.write("No proper arguments " % sys.argv[0])
		sys.exit(1)
	if len(sys.argv) > 1:
		hashmethod = sys.argv[1]
		if hashmethod == 'ahash':
			hashfunc = average_hash
		elif hashmethod == 'phash':
			hashfunc = phash
		elif hashmethod == 'dhash':
			hashfunc = dhash
		else:
			usage()
	else:
		usage()
	userpath = sys.argv[2] if len(sys.argv) > 2 else "."
	count = find_similar_images(userpath=userpath, hashfunc=hashfunc)
	print("n.of similar image sets are " + str(count))
        
