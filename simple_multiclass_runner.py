from helpers.dataloader import DataLoader
from helpers.model_trainer import ModelTrainer
from models.basic_model import Basic
import pandas as pd


def get_two_class_labels(csv_file_path, stratify_percentage=1):
    """returns a list of tuples where the first value is the file id and the second is the label
    [('ID_00019828f', 0)]
    """

    input_dataframe = pd.read_csv(csv_file_path)
    # filtered_input_dataframe = input_dataframe[input_dataframe['ID'].apply(lambda x : 'any' in x) ]
    files_with_ids = []

    # print(input_dataframe.columns.values)
    X = list(input_dataframe['ID'])
    y_dataframe = input_dataframe.drop(input_dataframe.columns[[0, 1, 7]], axis=1)

    # print(y_dataframe.head)

    # y = [y_dataframe.columns.values.tolist()] + y_dataframe.values.tolist()
    y = y_dataframe.values.tolist()
    # print (y[0])
    # print(len(X))
    # print(len(y))

    num_samples = int(stratify_percentage * len(X))
    print("Num Samples :", num_samples)

    for k, v in list(zip(X, y)):
        files_with_ids.append(("_".join(k.split('_')[:2]), v))

    return files_with_ids


def get_two_class_labels_fortest(csv_file_path_test, stratify_percentage=1):
    """returns a list of tuples where the first value is the file id and the second is the label
    [('ID_00019828f', 0)]
    """

    test_dataframe = pd.read_csv(csv_file_path_test)
    # filtered_input_dataframe = input_dataframe[input_dataframe['ID'].apply(lambda x : 'any' in x) ]
    files_with_ids_fortest = []

    # print(input_dataframe.columns.values)
    X_test = list(test_dataframe['ID'])

    print("Testing sample", X_test[0])

    y_test_df = test_dataframe.drop(test_dataframe.columns[[0, 6]], axis=1)

    print("testing y samples")

    # y = [y_dataframe.columns.values.tolist()] + y_dataframe.values.tolist()
    y_test = y_test_df.values.tolist()
    # print (y[0])
    # print(len(X))
    # print(len(y))

    num_samples_test = int(stratify_percentage * len(X_test))
    print("Num Samples in Testing :", num_samples_test)

    for k, v in list(zip(X_test, y_test)):
        files_with_ids_fortest.append(("_".join(k.split('_')[:2]), v))

    return files_with_ids_fortest

epoch_number = 5
input_filepath = "../../rsna-intracranial-hemorrhage-detection/"
train_image_filepath = "../../rsna-intracranial-hemorrhage-detection/stage_2_train/"

# Paths to csv files
csv_file_path = "./CSV/down_sampled_positive_data.csv"
csv_file_path_test = "./CSV/hem_positive_test_set.csv"
image_folder_root = train_image_filepath
files_with_ids = get_two_class_labels(csv_file_path,stratify_percentage=1)
X,y = [ x for x,y in files_with_ids], [y for x,y in files_with_ids]
print (len(files_with_ids))
print((y[0]))
print(X[0])

files_with_ids_fortest = get_two_class_labels_fortest(csv_file_path_test,stratify_percentage=1)
X_test,y_test = [ x_test for x_test,y_test in files_with_ids_fortest], [y_test for x_test,y_test in files_with_ids_fortest]
print (len(files_with_ids_fortest))
#print((y_test))
dataloader = DataLoader(train_image_filepath)
model = Basic(5,5)

trainer = ModelTrainer(dataloader,split_size=700)
model = trainer.fit(X,y,model,epochs = epoch_number)

prediction , accuracy,recall,precision,class_recall, class_precision  = trainer.predict(X_test,y_test,model)

print(prediction[1])
print(y_test[1])

print("Recall ",recall)
print("Accuracy ",accuracy)

model.model.summary()
