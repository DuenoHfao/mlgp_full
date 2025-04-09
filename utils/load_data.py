import os
import cv2

class LoadDataset:        
    def __getitem__(self, index):
        if index >= len(self.img_list):
            raise IndexError("Index out of range")
        
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image at {img_path} could not be loaded")
        return img
    
    def __len__(self):
        return len(self.img_list)
        
    def load_dataset(self, set_path=r'static\\assets'):

        sub_categories = os.listdir(set_path)
        img_list = []
        for category in sub_categories:
            category_list = []
            category_path = os.path.join(set_path, category)
            if not os.path.isdir(category_path):
                print(f"{set_path} is not a directory")
                continue

            images = os.listdir(category_path)
            for image in images:
                image_path = os.path.join(category_path, image)
                if not os.path.isfile(image_path):
                    print(f"{image_path} is not a file")
                    continue

                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: {image_path} is not a valid image file.")
                    continue
                
                cv2.destroyAllWindows()

                category_list.append(image_path)

            img_list.append(category_list)
        
        self.img_list = img_list
        return img_list

if __name__ == "__main__":
    data_set = LoadDataset()
    data_set.load_dataset()