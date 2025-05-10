import os
import pickle
from torch.utils import data
import tempfile
from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet


# def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
#     data_dir = os.path.join('./dataset/', dataset)
#     dataset_list = os.path.join(dataset, '_trainval_list.txt')
#     train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '_list.txt')
#     val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
#     inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

#     # inform_data_file collect the information of mean, std and weigth_class
#     if not os.path.isfile(inform_data_file):
#         print("%s is not found" % (inform_data_file))
#         dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
#                                                 inform_data_file=inform_data_file)

#         datas = dataCollect.collectDataAndSave()
#         if datas is None:
#             print("error while pickling data. Please check.")
#             exit(-1)
#     else:
#         print("find file: ", str(inform_data_file))
#         datas = pickle.load(open(inform_data_file, "rb"))


#     trainLoader = data.DataLoader(
#             CityscapesDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
#                               mirror=random_mirror, mean=datas['mean']),
#             batch_size=batch_size, shuffle=True, num_workers=num_workers,
#             pin_memory=True, drop_last=True)

#     valLoader = data.DataLoader(
#             CityscapesValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean']),
#             batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
#             drop_last=True)

#     return datas, trainLoader, valLoader

def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers, num_train, num_val):
    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = os.path.join(data_dir, dataset + '_trainval_list.txt')
    train_data_list_path = os.path.join(data_dir, dataset + '_' + train_type + '_list.txt')
    val_data_list_path = os.path.join(data_dir, dataset + '_val_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # Ensure inform_data_file exists
    if not os.path.isfile(inform_data_file):
        print(f"{inform_data_file} is not found")
        dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list, inform_data_file=inform_data_file)
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("Error while pickling data. Please check.")
            exit(-1)
    else:
        print("Found file:", inform_data_file)
        datas = pickle.load(open(inform_data_file, "rb"))

    # Read the original training and validation lists
    with open(train_data_list_path, "r") as f:
        train_lines = f.readlines()

    with open(val_data_list_path, "r") as f:
        val_lines = f.readlines()

    # Limit dataset size based on input
    train_lines = train_lines[:num_train]
    val_lines = val_lines[:num_val]

    # Create temporary files to store filtered lists
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as train_temp:
        train_temp.writelines(train_lines)
        train_temp_path = train_temp.name

    with tempfile.NamedTemporaryFile(delete=False, mode='w') as val_temp:
        val_temp.writelines(val_lines)
        val_temp_path = val_temp.name

    # Create data loaders
    trainLoader = data.DataLoader(
        CityscapesDataSet(data_dir, train_temp_path, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True)

    valLoader = data.DataLoader(
        CityscapesValDataSet(data_dir, val_temp_path, f_scale=1, mean=datas['mean']),
        batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
        drop_last=True)

    return datas, trainLoader, valLoader

def build_dataset_test(dataset, num_workers, num_test, none_gt=False):
    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = os.path.join(data_dir, dataset + '_trainval_list.txt')
    test_data_list_path = os.path.join(data_dir, dataset + '_test_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # Ensure inform_data_file exists
    if not os.path.isfile(inform_data_file):
        print(f"{inform_data_file} is not found")
        dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list, inform_data_file=inform_data_file)
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("Error while pickling data. Please check.")
            exit(-1)
    else:
        print("Found file:", inform_data_file)
        datas = pickle.load(open(inform_data_file, "rb"))

    # Choose correct dataset list based on `none_gt`
    if none_gt:
        with open(test_data_list_path, "r") as f:
            test_lines = f.readlines()
    else:
        test_data_list_path = os.path.join(data_dir, dataset + '_val_list.txt')
        with open(test_data_list_path, "r") as f:
            test_lines = f.readlines()

    # Limit dataset size based on input
    test_lines = test_lines[:num_test]

    # Create temporary file to store filtered list
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as test_temp:
        test_temp.writelines(test_lines)
        test_temp_path = test_temp.name

    # Create DataLoader
    if none_gt:
        testLoader = data.DataLoader(
            CityscapesTestDataSet(data_dir, test_temp_path, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        testLoader = data.DataLoader(
            CityscapesValDataSet(data_dir, test_temp_path, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return datas, testLoader