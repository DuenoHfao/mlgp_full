import os, shutil
import re
import random

def generate_train_val_test(src_path, train_path, val_path, test_path, split_ratio, seed=69420):
    if not os.path.exists(train_path):
        os.makedirs(train_path, exist_ok=True)
    if not os.path.exists(val_path):
        os.makedirs(val_path, exist_ok=True)
    if not os.path.exists(test_path):
        os.makedirs(test_path, exist_ok=True)

    if os.listdir(train_path) != []:
        shutil.rmtree(train_path)
        print(f"Removed existing train path: {train_path}")
    if os.listdir(val_path) != []:
        shutil.rmtree(val_path)
        print(f"Removed existing val path: {val_path}")
    if os.listdir(test_path) != []:
        shutil.rmtree(test_path)
        print(f"Removed existing test path: {test_path}")

    random.seed(seed)
    total_files = 0
    file_lists = []
    for (root, dirs, file_names) in os.walk(src_path):
        if file_names == []:
            continue

        sub_folder_name = os.path.basename(root)
        total_files += len(file_names)

        file_path_list = list(map(lambda path: os.path.join(sub_folder_name, path), file_names))

        random.shuffle(file_path_list)
        file_lists.append(file_path_list)

    for category_specific_list in file_lists:
        for file_name in category_specific_list:
            random_num = random.random()

            if random_num < split_ratio[0]:
                move_path = os.path.join(train_path, os.path.dirname(file_name))
            elif random_num < sum(split_ratio[:2]):
                move_path = os.path.join(val_path, os.path.dirname(file_name)) 
            else:
                move_path = os.path.join(test_path, os.path.dirname(file_name))

            os.makedirs(move_path, exist_ok=True)
            shutil.copy(os.path.join(src_path, file_name), move_path)

if __name__ == "__main__":
    generate_train_val_test(
        src_path='data/img_dataset',
        train_path='data/train_data',
        val_path='data/val_data',
        test_path='data/test_data',
        split_ratio=[0.7, 0.2, 0.1],
        seed=69420
    )