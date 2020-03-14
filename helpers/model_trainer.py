import gc
#import psutil
from cachetools import TTLCache
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle


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
                batch_imgs = []
                batch_labels = []

                current_x = X[processed:min(processed + self.split_size, len(y))]
                current_y = y[processed:min(processed + self.split_size, len(y))]
                for img, label in zip(current_x, current_y):
                    image = self.dataloader.load_image(img)
                    ##Figure out how many images are getting ignored because of this assumption
                    ##check if all reshape operations can happen in the dataloader
                    if image.shape != (512, 512):
                        continue
                    batch_imgs.append(image)
                    batch_labels.append(label)
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

    def predict(self, X_test, y_test, model, epochs=5, training_batch_size=8):

        splits = len(y) // self.split_size + 1
        splitter = StratifiedKFold(n_splits=splits, random_state=None, shuffle=False)
        processed = 0
        while processed < len(y_test):
            batch_imgs_test = []
            batch_labels_test = []

            current_x_test = X_test[processed:min(processed + self.split_size, len(y_test))]
            current_y_test = y_test[processed:min(processed + self.split_size, len(y_test))]
            for img, label in zip(current_x_test, current_y_test):
                image = self.dataloader.load_image(img)
                ##Figure out how many images are getting ignored because of this assumption
                ##check if all reshape operations can happen in the dataloader
                if image.shape != (512, 512):
                    continue
                batch_imgs_test.append(image)
                batch_labels_test.append(label)
            # print("Length of images",len(batch_imgs_test))
            # print("using  batch with size", len(batch_imgs_test), len(batch_labels_test), "Processed ", processed, "Total ", len(y_test))

            batch_imgs_test = np.array(batch_imgs_test)
            # print("Length of images",len(batch_imgs_test))
            # print("Shape before reshaping",batch_imgs_test.shape)

            # predict_input = batch_imgs_test.reshape(len(X_test),2)

            # batch_imgs_test = np.expand_dims(batch_imgs_test, axis=1)
            # batch_imgs_test = np.expand_dims(batch_imgs_test, axis=1)

            # print("shape of input to predict",batch_imgs_test.shape)
            preds = model.predict(batch_imgs_test)
            # print("the predicted y is of length : ", len(preds))
            # print("first sample prediciton is ", preds[0])
            ##classwise precision, recall
            ##classwise precision, recall
            acc_sum = 0
            true_pos = 0
            false_pos = 0
            true_neg = 0
            false_neg = 0

            class_true_pos = np.zeros([5, 1])
            class_true_neg = np.zeros([5, 1])
            class_false_pos = np.zeros([5, 1])
            class_false_neg = np.zeros([5, 1])

            class_recall = np.zeros([5, 1])
            class_precision = np.zeros([5, 1])

            for i in range(len(preds)):
                for j in range(len(preds[0])):
                    if (preds[i][j]) >= 0.5:
                        preds[i][j] = 1
                    if (preds[i][j]) < 0.5:
                        preds[i][j] = 0
            for a in range(len(preds)):
                if (np.all(batch_labels_test[a] == preds[a])):
                    acc_sum = acc_sum + 1;
                for b in range(len(preds[a])):
                    if (batch_labels_test[a][b] == preds[a][b] and preds[a][b] == 1):
                        # acc_sum = acc_sum+1
                        class_true_pos[b] = class_true_pos[b] + 1
                        true_pos = true_pos + 1

                    if (batch_labels_test[a][b] == preds[a][b] and preds[a][b] == 0):
                        class_true_neg[b] = class_true_neg[b] + 1
                        true_neg = true_neg + 1

                    if (batch_labels_test[a][b] < preds[a][b]):
                        class_false_pos[b] = class_false_pos[b] + 1
                        false_pos = false_pos + 1

                    if (batch_labels_test[a][b] > preds[a][b]):
                        class_false_neg[b] = class_false_neg[b] + 1
                        false_neg = false_neg + 1

            accuracy = (acc_sum / len(preds)) * 100
            recall = (true_pos) / (true_pos + false_neg)
            precision = (true_pos) / (true_pos + false_pos)

            for c in range(len(class_recall)):
                class_recall[c] = (class_true_pos[c]) / (class_true_pos[c] + class_false_neg[c])
                class_precision[c] = (class_true_pos[c]) / (class_true_pos[c] + class_false_pos[c])

            print("Accuracy", accuracy)
            print("Recall", recall)
            print("Precision", precision)
            print("Class-wise precision\n", class_precision)
            print("Class-wise recall\n", class_recall)
            print("\n")

            self.dataloader.trigger_expire()
            del batch_imgs_test
            del batch_labels_test
            processed += self.split_size

        return (preds, accuracy, recall, precision, class_recall, class_precision)