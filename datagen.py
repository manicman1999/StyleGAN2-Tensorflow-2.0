from PIL import Image
import numpy as np
import random
import os

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r %s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()
        print()






class dataGenerator(object):

    def __init__(self, folder, im_size, mss = (1024 ** 3), flip = True, verbose = True):
        self.folder = folder
        self.im_size = im_size
        self.segment_length = mss // (im_size * im_size * 3)
        self.flip = flip
        self.verbose = verbose

        self.segments = []
        self.images = []
        self.update = 0

        if self.verbose:
            print("Importing images...")
            print("Maximum Segment Size: ", self.segment_length)

        try:
            os.mkdir("data/" + self.folder + "-npy-" + str(self.im_size))
        except:
            self.load_from_npy(folder)
            return

        self.folder_to_npy(self.folder)
        self.load_from_npy(self.folder)

    def folder_to_npy(self, folder):

        if self.verbose:
            print("Converting from images to numpy files...")

        names = []

        for dirpath, dirnames, filenames in os.walk("data/" + folder):
            for filename in [f for f in filenames if (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".JPEG"))]:
                fname = os.path.join(dirpath, filename)
                names.append(fname)

        np.random.shuffle(names)

        if self.verbose:
            print(str(len(names)) + " images.")



        kn = 0
        sn = 0

        segment = []

        for fname in names:
            if self.verbose:
                print('\r' + str(sn) + " // " + str(kn) + "\t", end = '\r')

            try:
                temp = Image.open(fname).convert('RGB').resize((self.im_size, self.im_size), Image.BILINEAR)
            except:
                print("Importing image failed on", fname)
            temp = np.array(temp, dtype='uint8')
            segment.append(temp)
            kn = kn + 1

            if kn >= self.segment_length:
                np.save("data/" + folder + "-npy-" + str(self.im_size) + "/data-"+str(sn)+".npy", np.array(segment))

                segment = []
                kn = 0
                sn = sn + 1


        np.save("data/" + folder + "-npy-" + str(self.im_size) + "/data-"+str(sn)+".npy", np.array(segment))


    def load_from_npy(self, folder):

        for dirpath, dirnames, filenames in os.walk("data/" + folder + "-npy-" + str(self.im_size)):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                self.segments.append(os.path.join(dirpath, filename))

        self.load_segment()

    def load_segment(self):

        if self.verbose:
            print("Loading segment")

        segment_num = random.randint(0, len(self.segments) - 1)

        self.images = np.load(self.segments[segment_num])

        self.update = 0

    def get_batch(self, num):

        if self.update > self.images.shape[0]:
            self.load_from_npy(self.folder)

        self.update = self.update + num

        idx = np.random.randint(0, self.images.shape[0] - 1, num)
        out = []

        for i in idx:
            out.append(self.images[i])
            if self.flip and random.random() < 0.5:
                out[-1] = np.flip(out[-1], 1)

        return np.array(out).astype('float32') / 255.0


