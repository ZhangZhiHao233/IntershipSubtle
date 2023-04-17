from torch import optim
from torch.utils.data import Dataset
from utils import *
from dataset import *
from networks import *
import matplotlib.pyplot as plt
import argparse
import numpy as np
import wandb

model_path = 'checkpoint'
if not os.path.exists(model_path):
    os.mkdir(model_path)

log_path = 'log'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logger = get_logger(log_path)

mask_path = 'masks'
if not os.path.exists(mask_path):
    os.mkdir(mask_path)

dataset_path = '../../input_path/dataset_pd_fs.npz'
device = torch.device("cuda:1")
CHECKPOINT_PATH_WANDB = './latest.pth'
my_yaml_file = './config-defaults.yaml'
EPOCHES = 300
BATCHSIZE = 4

dataset_train = CreateDataset_npz(phase="train", dataset_path=dataset_path)
data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                          batch_size=BATCHSIZE,
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=True,
                                          sampler=None,
                                          drop_last=True)

dataset_test = CreateDataset_npz(phase="test", dataset_path=dataset_path)
data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1,
                                          pin_memory=True,
                                          sampler=None,
                                          drop_last=True)

criterion_g = MixedPix2PixLoss(alpha=0.5).to(device)
set_seed_torch(14)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(img1.max() / torch.sqrt(mse))

to_range_0_1 = lambda x: (x + 1.) / 2.

def train(data_loader_train, epoch):

    loss_gbs = []
    gb.train()

    for iteration, (samp_input, samp_pd) in enumerate(data_loader_train):
        optimizer_gb.zero_grad()

        if iteration % 8 != 0:
            continue
        (samp_fs, _) = torch.split(samp_input, [1, 1], dim=1)
        samp_fs_gpu = samp_fs.to(device)
        samp_pd_gpu = samp_pd.to(device)
        out_global = gb(samp_fs_gpu)

        loss_gb = criterion_g(out_global, samp_pd_gpu)
        loss_gbs.append(loss_gb.item())
        loss_gb.backward()
        optimizer_gb.step()

    loss_gbs_v = np.sum(loss_gbs)

    #wandb.log({"train loss": loss_gbs_v})

    print('train epoch:', epoch, 'loss: ', loss_gbs_v)
    torch.save({  # Save our checkpoint loc
        'epoch': epoch,
        'model_state_dict': gb.state_dict(),
        'optimizer_state_dict': optimizer_gb.state_dict(),
        'loss': loss_gbs_v,
    }, os.path.join(model_path, 'model_{}.pth'.format(epoch)))

    torch.save({  # Save our checkpoint loc
        'epoch': epoch,
        'model_state_dict': gb.state_dict(),
        'optimizer_state_dict': optimizer_gb.state_dict(),
        'loss': loss_gbs_v,
    }, CHECKPOINT_PATH_WANDB)
    wandb.save(CHECKPOINT_PATH_WANDB, policy='end')  # saves checkpoint to wandb

    print('** save epoch ', epoch, '.')
    log_str = " train epoch:%d loss_gb:%f" % (epoch, loss_gbs_v)
    logger.info(log_str)
    return loss_gbs_v


def test(data_loader_test, epoch):
    loss_gbs = []
    gb.eval()

    psnrs_g = np.zeros((1, len(data_loader_test)))

    for iteration, (samp_input, samp_pd) in enumerate(data_loader_test):
        optimizer_gb.zero_grad()

        if iteration % 3 == 0:
            continue

        (samp_fs, _) = torch.split(samp_input, [1, 1], dim=1)
        samp_fs = samp_fs.to(device)
        samp_pd = samp_pd.to(device)
        out_global = gb(samp_fs)

        loss_gb = criterion_g(out_global, samp_pd)
        loss_gbs.append(loss_gb.item())

        # if iteration == len(data_loader_test)//3:
        # if iteration % 4 == 0:
        #samp_fs_0 = samp_fs[0].detach().cpu().squeeze().numpy()
        #outputg_0 = out_global[0].detach().cpu().squeeze().numpy()
        #samp_pd_0 = samp_pd[0].detach().cpu().squeeze().numpy()

        #plt.figure(figsize=(9, 3), dpi=300, tight_layout=True)
        #plt.subplot(1, 3, 1)
        #plt.imshow(samp_fs_0, cmap="gray")
        #plt.subplot(1, 3, 2)
        #plt.imshow(outputg_0, cmap="gray")
        #plt.subplot(1, 3, 3)
        #plt.imshow(samp_pd_0, cmap="gray")
        # plt.show()
        #plt.savefig("masks/epoch{}_{}.jpg".format(epoch, iteration), dpi=600)
        #plt.clf()
        #plt.close()

        #print('save png.. iteration', iteration)

        out_g = to_range_0_1(out_global)
        samp_pd = to_range_0_1(samp_pd)

        psnrs_g[0, iteration] = psnr(out_g, samp_pd).detach().cpu().numpy()

    #wandb.log({"test loss": np.sum(loss_gbs)})
    print('test epoch:', epoch, 'loss_gb: ', np.sum(loss_gbs))
    print('psnr_total:', np.nanmean(psnrs_g))
    log_str = " test epoch:%d loss_gb:%f psnr_total:%f" % (epoch, np.sum(loss_gbs), np.nanmean(psnrs_g))
    logger.info(log_str)
    return np.sum(loss_gbs), np.nanmean(psnrs_g)

gb = AttU_Net(1, 1).to(device)
optimizer_gb = optim.Adam(gb.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0.0001)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     "-b",
    #     "--batch_size",
    #     type=int,
    #     default=32,
    #     help="Batch size")
    # parser.add_argument(
    #     "-e",
    #     "--epochs",
    #     type=int,
    #     default=50,
    #     help="Number of training epochs")
    # parser.add_argument(
    #     "-lr",
    #     "--learning_rate",
    #     type=int,
    #     default=0.001,
    #     help="Learning rate")

    # hyperparameter_defaults = dict(
    #     dropout=0.5,
    #     batch_size=100,
    #     learning_rate=0.001,
    # )

    config_dictionary = dict(
        yaml=my_yaml_file
        # params=hyperparameter_defaults,
    )
    wandb.init(config=config_dictionary)
    print(wandb.config['only test'])
    # resume = True
    # config = {
    #     "epochs": EPOCHES,
    #     "learning_rate": 0.0001,
    #     "batch_size": BATCHSIZE
    # }
    #
    # run = wandb.init(
    #     project="test_wandb",
    #     group="experiment_1",
    #     notes="My first experiment",
    #     tags=["baseline", "unet1"],
    #     config=config,
    #     resume=resume
    # )
    # last_epoch=0
    # if wandb.run.resumed:
    #     checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH_WANDB))
    #     gb.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer_gb.load_state_dict(checkpoint['optimizer_state_dict'])
    #     last_epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']
    #
    #     print("resume from epoch%d, loss=%f" % (last_epoch, loss))

    # if os.path.exists(model_path):
    #     dir_list = os.listdir(model_path)
    #     if len(dir_list) > 0:
    #         dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
    #         print('dir_list: ', dir_list)
    #
    #         last_model_name = model_path + '/' + dir_list[-1]
    #         checkpoint = torch.load(last_model_name)
    #
    #         gb.load_state_dict(checkpoint['model_gb'])
    #         last_epoch = checkpoint['epoch']
    #         loss_gb = checkpoint['loss_gb']
    #
    #         print('load epoch {} succeed!'.format(last_epoch))
    #         print('loss_gb {}'.format(loss_gb))
    #
    #     else:
    #         last_epoch = 0
    #         print('no saved model, start a new train.')
    #
    # else:
    #     last_epoch = 0
    #     print('no saved model, start a new train.')

    for epoch in range(last_epoch + 1, EPOCHES + 1):
        train_loss = train(data_loader_train, epoch)
        #if epoch % 5 == 0:
        test_loss, test_psnr = test(data_loader_test, epoch)
        #break
        wandb.log({"train loss": train_loss, "test loss": test_loss, "test_psnr":test_psnr})



