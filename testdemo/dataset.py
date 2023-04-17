import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import random

MASK_POINT = []
MASK_AREA = []
def generate_mask(img_size, val=False):
    mask = np.ones((img_size, img_size))
    mask_size = random.randint(img_size//4, img_size-1)
    # mask_size = random.randint(img_size//16, img_size//4)

    if mask_size == 0:
        return mask
    elif mask_size == img_size-1 or val==True:
        return 1-mask

    c_x = random.randint(0, img_size-1)
    c_y = random.randint(0, img_size-1)

    box_l_x = c_x-mask_size//2
    box_l_y = c_y-mask_size//2
    box_r_x = c_x+mask_size//2
    box_r_y = c_y+mask_size//2

    if box_l_x < 0:
        box_l_x = 0
    if box_l_y < 0:
        box_l_y = 0
    if box_r_x > img_size-1:
        box_r_x = img_size-1
    if box_r_y > img_size-1:
        box_r_y = img_size-1

    mask[box_l_y:box_r_y, box_l_x:box_r_x] = 0
    # print('*', c_y-mask_size//2, c_y + mask_size, c_x-mask_size//2,c_x + mask_size)

    MASK_POINT.append([c_x, c_y])
    MASK_AREA.append([box_l_y, box_r_y, box_l_x, box_r_x])
    return mask

def getmsks(img_size, channels, val=False):
    msks = []
    for i in range(channels):
        msks.append(generate_mask(img_size, val))
    msks = np.array(msks)
    msks = msks[:, np.newaxis, :, :]
    return msks

def CreateDataset_npz(phase, dataset_path):
    ds = np.load(dataset_path)
    train_fs = ds['train_fs']
    train_pd = ds['train_pd']
    test_fs = ds['test_fs']
    test_pd = ds['test_pd']

    if phase == 'train':
        train_fs = LoadDataSet_npz(train_fs)
        train_pd = LoadDataSet_npz(train_pd)
        masks = getmsks(train_fs.shape[2], train_fs.shape[0])
        masks = masks.astype(np.float32)
        input = np.concatenate((train_fs, masks), axis=1)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(input), torch.from_numpy(train_pd))
        return dataset
    else:
        test_fs = LoadDataSet_npz(test_fs)
        test_pd = LoadDataSet_npz(test_pd)
        masks = getmsks(test_fs.shape[2], test_fs.shape[0], True)
        masks = masks.astype(np.float32)
        input = np.concatenate((test_fs, masks), axis=1)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(input), torch.from_numpy(test_pd))
        return dataset

def LoadDataSet_npz(data):
    data = data.astype(np.float32)
    data = np.expand_dims(data, axis=1)
    data = (data - 0.5) / 0.5
    print(np.min(data), np.max(data))
    return data


if __name__ == '__main__':


    dataset = CreateDataset_npz(phase="train", dataset_path="../input_path_288/dataset_pd_fs.npz")
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=3,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=None,
                                              drop_last=True)

    for iteration, (x_val, y_val) in enumerate(data_loader):

        print(x_val.shape, y_val.shape)

        plt.figure(figsize=(9, 9), dpi=300, tight_layout=True)
        plt.subplot(3, 3, 1)
        plt.imshow(x_val[0][0].squeeze(), cmap="gray")
        plt.subplot(3, 3, 2)
        plt.imshow(x_val[1][0].squeeze(), cmap="gray")
        plt.subplot(3, 3, 3)
        plt.imshow(x_val[2][0].squeeze(), cmap="gray")

        plt.subplot(3, 3, 4)
        plt.imshow(x_val[0][1].squeeze()*x_val[0][0].squeeze(), cmap="gray")
        plt.subplot(3, 3, 5)
        plt.imshow(x_val[1][1].squeeze()*x_val[1][0].squeeze(), cmap="gray")
        plt.subplot(3, 3, 6)
        plt.imshow(x_val[2][1].squeeze()*x_val[2][0].squeeze(), cmap="gray")

        plt.subplot(3, 3, 7)
        plt.imshow(y_val[0].squeeze(), cmap="gray")
        plt.subplot(3, 3, 8)
        plt.imshow(y_val[1].squeeze(), cmap="gray")
        plt.subplot(3, 3, 9)
        plt.imshow(y_val[2].squeeze(), cmap="gray")

        plt.show()
        # plt.savefig("masks/{}.jpg".format(iteration), dpi=300)
        plt.clf()
        plt.close()

        # break

    # imgw = 256
    # imgh = 256
    #
    # for i in range(1000):
    #     print(i)
    #     mask = generate_mask(imgw)
    #
    #     plt.imshow(mask, cmap="gray")
    #     plt.show()
    #     plt.clf()
    #     plt.close()

    print(len(MASK_POINT), MASK_POINT)
    print(len(MASK_AREA), MASK_AREA)
    MASK_POINT = np.array(MASK_POINT)
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(MASK_POINT[:,0], MASK_POINT[:,1], s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, 272, 0, 272])
    plt.show()
    plt.clf()
    plt.close()

    mask = np.zeros([272, 272])
    for i in range(len(MASK_AREA)):
        mask[MASK_AREA[i][0]:MASK_AREA[i][1], MASK_AREA[i][2]:MASK_AREA[i][3]] += 0.1

    plt.imshow(mask)
    plt.colorbar()
    plt.show()