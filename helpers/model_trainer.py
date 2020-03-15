import gc
#import psutil
from cachetools import TTLCache
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from helpers.metric import Metric
import numpy as np
from joblib import Parallel,delayed


class ModelTrainer(object):

    def __init__(self, dataloader, training_batch_size=8, split_size=400):

        self.dataloader = dataloader
        self.split_size = split_size

    """Takes X and y as the file name and labels and """

    def fit(self, X, y, model, epochs=5, training_batch_size=8):
        splits = len(y) // self.split_size + 1

        splitter = StratifiedKFold(n_splits=splits, random_state=None, shuffle=True)
        count = 1
        while epochs > 0:
            X, y = shuffle(X, y)
            ##TODO: add a better split and shuffle mechanism
            processed = 0
            while processed < len(y):


                current_x = X[processed:min(processed + self.split_size, len(y))]
                current_y = y[processed:min(processed + self.split_size, len(y))]
                batch_imgs, batch_labels = self.load_images(current_x, current_y)
                print("Length of images", len(batch_imgs))
                print("Using  batch with size", len(batch_imgs), len(batch_labels), "Processed ", processed, "Total ",
                      len(y))
                model.fit(np.array(batch_imgs), np.array(batch_labels))
                # model.predict(np.array(batch_imgs),np.array(batch_labels))
                self.dataloader.trigger_expire()
                del batch_imgs
                del batch_labels

                gc.collect()
                processed += self.split_size
            print("Ending epoch", count)
            # save_model(model.model,"epoch-model-four.hdf5")
            # model.save("basic-model-1-epochs.h5")
            # model.model.save_model(model,h5pyBasic-model-1-epoch)
            epochs -= 1
            count += 1

        return model

    def load_images(self, current_x, current_y, exceptions = set(["ID_6431af929"])):
        batch_imgs = []
        batch_labels = []
        img_label_pairs = [ (x, y) for x,y in zip(current_x,current_y) if x not in exceptions ]
        # for img, label in img_label_pairs:
        #     image = self.dataloader.load_image(img)
        #     ##Figure out how many images are getting ignored because of this assumption
        #     ##check if all reshape operations can happen in the dataloader
        #     if image is None or image.shape != (512, 512):
        #         continue
        #     batch_imgs.append(image)
        #     batch_labels.append(label)
        img_label_pairs = Parallel(n_jobs=-1)(delayed(self.load_image)(k,v) for k,v in img_label_pairs)
        batch_imgs = [k for k,v in img_label_pairs if k is not None and k.shape == (512,512)]
        batch_labels = [v for k,v in img_label_pairs if k is not None and k.shape == (512,512)]

        return batch_imgs,batch_labels

    def load_image(self, img_name,label):
        return self.dataloader.load_image(img_name), label

    def predict(self, X_test, y_test, model, epochs=5, training_batch_size=8):
        print("Predicting")

        splits = len(y_test) // self.split_size + 1
        splitter = StratifiedKFold(n_splits=splits, random_state=None, shuffle=False)
        processed = 0
        class_wise_metrics = dict()
        overall_test_sample_count = 0
        overall_correct_classifications = 0

        while processed < len(y_test):
            batch_imgs_test = []
            batch_labels_test = []

            current_x_test = X_test[processed:min(processed + self.split_size, len(y_test))]
            current_y_test = y_test[processed:min(processed + self.split_size, len(y_test))]
            for img, label in zip(current_x_test, current_y_test):
                image = self.dataloader.load_image(img)
                if image.shape != (512, 512):
                    continue
                batch_imgs_test.append(image)
                batch_labels_test.append(label)
            batch_imgs_test = np.array(batch_imgs_test)
            preds = model.predict(batch_imgs_test)

            for i in range(len(preds)):
                for j in range(len(preds[0])):
                    if (preds[i][j]) >= 0.5:
                        preds[i][j] = 1
                    if (preds[i][j]) < 0.5:
                        preds[i][j] = 0
            for a in range(len(preds)):
                overall_test_sample_count+=1
                if (np.all(batch_labels_test[a] == preds[a])):
                    overall_correct_classifications+=1
                for b in range(len(preds[a])):
                    if b not in class_wise_metrics:
                        class_wise_metrics[b] = Metric()

                    if (batch_labels_test[a][b] == preds[a][b] and preds[a][b] == 1):
                        class_wise_metrics[b].add_true_positive()

                    if (batch_labels_test[a][b] == preds[a][b] and preds[a][b] == 0):

                        class_wise_metrics[b].add_true_negative()

                    if (batch_labels_test[a][b] < preds[a][b]):

                        class_wise_metrics[b].add_false_positive()

                    if (batch_labels_test[a][b] > preds[a][b]):

                        class_wise_metrics[b].add_false_negative()

            self.dataloader.trigger_expire()
            del batch_imgs_test
            del batch_labels_test
            processed += self.split_size
            print (class_wise_metrics)
        print ("Overall class metrics")
        for k,v in class_wise_metrics.items():
            print ("Class {0}: {1}".format(k,v))
        print ("Overall accuracy", 100*overall_correct_classifications/overall_test_sample_count)
        return class_wise_metrics, overall_correct_classifications,overall_test_sample_count

