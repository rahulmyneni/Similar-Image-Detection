import os,sys
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

def is_image(filename):
        f = filename.lower()
        return f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".bmp") or f.endswith(".gif")

def hash_diff(im, userpath, hashfunc):
    image_filenames = [os.path.join(userpath, file) for file in os.listdir(userpath) if is_image(file)]
    hash1 = hashfunc(im)
    img_list = []
    img_count = 0
    error = 0
    for img in image_filenames:
        hash = hashfunc(Image.open(img))
        ham = hd(str(hash), str(hash1))
        if ham < 5:
            img_list.append(img)
    if len(img_list) > 1:
        sys.stderr.write("Found similar diagrams")
        sys.stderr.write("\n")
        sys.stderr.write(" ".join(img_list))
        sys.stderr.write("\n")
        img_count += 1
    else:
        sys.stderr.write("No similar diagram")
        sys.stderr.write(similar_name)
        error += 1
    return img_count, error

if __name__ == '__main__':
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

    sim = 0
    notsim = 0
    userpath = sys.argv[2] if len(sys.argv) > 2 else "."
    similar_path = 'C:/Users/krishnaprasad/Desktop/jpg/single'
    similar_images = [os.path.join(similar_path, file) for file in os.listdir(similar_path) if is_image(file)]
    total = len(similar_images)
    for similar_name in similar_images:
        im = Image.open(similar_name)
        count, error = hash_diff(im, userpath=userpath, hashfunc=hashfunc)
        sim += count
        notsim += error
    accu = sim/total
    print("Accuracy =" + str(accu))
    print(notsim)
