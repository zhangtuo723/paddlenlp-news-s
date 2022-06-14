from paddle.io import Dataset
import paddle
import pickle
class MyDataset(Dataset):
    def __init__(self,data_path):
        with open(data_path,"rb") as f:
            data_list = pickle.load(f)
        self.data = data_list

    def __getitem__(self,idx):
        return paddle.to_tensor([self.data[idx][0]]),paddle.to_tensor([self.data[idx][1]]),self.data[idx][2]
    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    train_path = "./data/train.pkl"
    mydataset = MyDataset(train_path)

    print(mydataset[0])
