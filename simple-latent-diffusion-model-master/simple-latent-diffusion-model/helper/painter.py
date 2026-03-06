import io
from PIL import Image as im
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np

class Painter(object):
    def __init__(self) :
        pass

    def show_images(self, images, title : str = '', index : bool = False, cmap = None, show = True):
        images = images.permute(0, 2, 3, 1) 
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()
        images = np.clip(images / 2 + 0.5, 0, 1)
        
        fig = plt.figure(figsize=(8, 8))
        rows = int(len(images) ** (1 / 2))
        cols = round(len(images) / rows)

        idx = 0
        for _ in range(rows):
            for _ in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx], cmap = cmap)
                    if index :
                        plt.title(idx + 1)
                    plt.axis('off')
                    idx += 1
        fig.suptitle(title, fontsize=30)
        if show:
            plt.show()

    def show_first_batch(self, loader):
        for batch in loader:
            self.show_images(images = batch, title = "First Batch")
            break
        
    def make_gif(self, images, file_name):
        imgs = []
        for i in tqdm(range(len(images))):
            img_buf = io.BytesIO()
            self.show_images(images[i], title = 't = ' + str(i), show=False)
            plt.savefig(img_buf, format='png')
            imgs.append(im.open(img_buf))
        imgs[0].save(file_name + '.gif', format='GIF', append_images=imgs, save_all=True, duration=1, loop=0)
        plt.close('all')
            