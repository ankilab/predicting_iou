import tensorflow as tf
import numpy as np
import imageio as io
import pandas as pd
import gc

class DataGenerator(tf.keras.utils.PyDataset):
    def __init__(self, fn, folder, dg="A", data_arrangement='see', endoscopy_im=True, batch_size=32, seed=42, **kwargs):
        super().__init__(**kwargs)

        # Save settings
        self.data = pd.read_csv(fn)
        self.folder = folder
        self.batch_size = batch_size
        self.data_arrangement = data_arrangement

        self.idxs = np.arange(len(self.data))
        
        # Ensure random seed for reproducibility reasons
        np.random.seed(seed)
        np.random.shuffle(self.idxs)

        if dg == "A":
            self.idxs = self.idxs[:int(0.95 * len(self.idxs))]
        else:
            self.idxs = self.idxs[-int(0.05 * len(self.idxs)):]
        
        self.endoscopy_im = endoscopy_im

    def __len__(self):
        return int(np.floor(len(self.idxs) / self.batch_size))

    def __getitem__(self, idx):
        idxs = self.idxs[idx*self.batch_size:(idx+1)*self.batch_size]

        X = []
        y = []

        # Iterate through images for a given batch
        for i in idxs:
            # Load the index and the target iou score
            idx = int(self.data.iloc[i]['i'])
            iou = self.data.iloc[i]['iou']

            # Open segmentation map with artifacts
            seg = io.v3.imread(self.folder+f"/{idx}_xseg.png")

            # endoscopy image should be part of the input data
            if self.endoscopy_im:
                end = io.v3.imread(self.folder+f"/{idx}.png")

                # Convert RGB to grayscale
                if end.shape[-1] == 3:
                    end = end @ [0.2989, 0.5870, 0.1140]

                # Do min/max scaling
                end = (end - end.min()) / (end.max() - end.min())

                to_X = []

                # add data according to the data format
                # s: add segmentation map
                # e: add endoscopy map
                for qi in self.data_arrangement:
                    to_X.append(seg[..., None] if qi=="s" else end[..., None])

                to_X = np.concatenate(to_X, axis=2)

            else:
                to_X = seg[..., None]

            # Add information to the batch
            X.append(to_X)
            y.append(iou)

        # Return batch as numpy array
        return np.asarray(X), \
            np.asarray(y)

    def on_epoch_end(self):
        # Shuffle IDs at end of epochs and explicitly do garbage collection
        np.random.shuffle(self.idxs)
        gc.collect()


if __name__ == '__main__':
    # Define the data foulder and the meta data file
    folder = 'training_224x224'
    fn = 'training_224x224_xseg_meta.csv'

    # Create datagenerator
    dg = DataGenerator(fn, folder, data_arrangement="ses")

    X, y = dg[0]

    # Show some samples
    import matplotlib.pyplot as plt

    i = 0

    for xi, yi in zip(X,y):
        plt.figure()
        plt.subplot(131)
        plt.imshow(xi[..., 0])

        plt.title(yi)
        plt.subplot(132)
        plt.imshow(xi[..., 1])

        plt.title(yi)
        plt.subplot(133)
        plt.imshow(xi[..., 2])


        plt.show()
        i += 1
        if i == 10:
            break