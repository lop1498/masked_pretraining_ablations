import matplotlib.pyplot as plt
import numpy as np
import pickle

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def load_file(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


dataset = "mrpc"
file_name_pretrained = "./results/{}_pretrainedBERT".format(dataset)
file_name_random = "./results/{}_randomBERT".format(dataset)
loss_pretrained = load_file(file_name_pretrained)
loss_random = load_file(file_name_random)
probs = np.arange(0,1,0.01)
plt.plot(probs, loss_pretrained, label='Pretrained BERT')
plt.plot(probs, loss_random, label='Random BERT')

plt.xlabel('Masking percentage')
plt.ylabel('Test loss')
plt.legend()

bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="gray", lw=0.5)
plt.text(0.05, 0.88, dataset, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=bbox_props)

plt.show()