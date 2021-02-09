import image
from imagehash import *

def hd(str1, str2):
    i = 0
    count = 0
    l = len(str1)
    while l >= 0:
        if str1[l-1] != str2[l-1]:
            count+=1
        l -= 1
    return count

def hash_diff(im, userpath, hashfunc):
    import os
    def is_image(filename):
        f = filename.lower()
        return f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".bmp") or f.endswith(".gif")
    image_filenames = [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
    hash1 = hashfunc(im)
    img_list = []
    for img in image_filenames:
        hash = hashfunc(Image.open(img))
        print(img)
        ham = hd(str(hash), str(hash1))
        print(ham)
        if ham < 5:
            img_list.append(img)
    if len(img_list) > 1:
        sys.stderr.write(" ".join(img_list))
        sys.stderr.write("\n")
    else:
        sys.stderr.write("No similar diagrams")


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
    im = Image.open("C:/Users/krishnaprasad/Desktop/jpg/similar/94 (1).jpg")
    userpath = sys.argv[2] if len(sys.argv) > 2 else "."
    count = hash_diff(im, userpath=userpath, hashfunc=hashfunc)
