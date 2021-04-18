import torch
import torch.utils.data as data
import numpy as np
import torchnet as tnt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import os
import json
import pickle as pkl
import argparse
import pprint

from models.stclassifier import PseTae
from dataset import PixelSetData, PixelSetData_preloaded
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis

import seaborn as sns
import matplotlib.pyplot as plt

def train_epoch(model, optimizer, criterion, data_loader, device, config):
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(data_loader):

        y_true.extend(list(map(int, y)))

        x = recursive_todevice(x, device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y.long())
        loss.backward()
        optimizer.step()

        pred = out.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}'.format(i + 1, len(data_loader), loss_meter.value()[0],
                                                                    acc_meter.value()[0]))

    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_accuracy': acc_meter.value()[0],
                     'train_IoU': mIou(y_true, y_pred, n_classes=config['num_classes'])}

    return epoch_metrics


def evaluation(model, criterion, loader, device, config, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, y) in loader:
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)
            loss = criterion(prediction, y)

        acc_meter.add(prediction, y)
        loss_meter.add(loss.item())

        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))

    metrics = {'{}_accuracy'.format(mode): acc_meter.value()[0],
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_IoU'.format(mode): mIou(y_true, y_pred, config['num_classes'])}

    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes'])))


def get_pse(folder, config):
    if config['preload']:
        dt = PixelSetData_preloaded(config[folder], labels='CODE_GROUP', npixel=config['npixel'],
                          sub_classes=[1, 2, 3, 4, 5, 8, 11, 16, 17, 18,19, 20, 24, 25],
                          #sub_classes = [1, 2, 3, 16, 17, 18],
                          norm=None,
                          sensor = config['sensor'],
                          extra_feature=None, 
                          jitter=None)
    else:
        dt = PixelSetData(config[folder], labels='CODE_GROUP', npixel=config['npixel'],
                          sub_classes=[1, 2, 3, 4, 5, 8, 11, 16, 17, 18,19, 20, 24, 25],
                          #sub_classes = [1, 2, 3, 16, 17, 18],
                          norm=None,
                          sensor = config['sensor'],
                          extra_feature=None, 
                          jitter=None)
    return dt


def get_loaders(config):
    loader_seq =[]
    train_loader = get_pse('dataset_folder', config)

    if config['dataset_folder2'] is not None:
        train_loader2 = get_pse('dataset_folder2', config)
        train_loader = data.ConcatDataset([train_loader, train_loader2])
        
    train_loader = data.DataLoader(train_loader, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'], shuffle =True)
    validation_loader = data.DataLoader(get_pse('val_folder', config), batch_size=config['batch_size'],
                                        num_workers=config['num_workers'], shuffle = False)

    test_loader = data.DataLoader(get_pse('test_folder', config), batch_size=config['batch_size'],
                                    num_workers=config['num_workers'], shuffle = False)

    loader_seq.append((train_loader, validation_loader, test_loader))
    return loader_seq

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)


def checkpoint(log, config):
    with open(os.path.join(config['res_dir'], 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, conf_mat, config):
    with open(os.path.join(config['res_dir'], 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'conf_mat.pkl'), 'wb'))

    # ----> save confusion matrix 
    plt.figure(figsize=(15,10))
    img = sns.heatmap(conf_mat, annot = True, fmt='d',linewidths=0.5, cmap='OrRd')
    img.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    img.set(ylabel="True Label", xlabel="Predicted Label")
    img.figure.savefig(os.path.join(config['res_dir'], 'conf_mat_picture.png'))
    img.get_figure().clf()


def overall_performance(config):
    cm = np.zeros((config['num_classes'], config['num_classes']))
    cm += pkl.load(open(os.path.join(config['res_dir'], 'conf_mat.pkl'), 'rb'))
    per_class, perf = confusion_matrix_analysis(cm)

    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))
    with open(os.path.join(config['res_dir'], 'per_class.json'), 'w') as file:
        file.write(json.dumps(per_class, indent=4))


def plot_metrics(config):
    with open(os.path.join(config['res_dir'], 'trainlog.json'), 'r') as file:
        d = json.loads(file.read())
    
    epoch = [i+1 for i in range(len(d))]
    train_loss = [d[str(i+1)]["train_loss"] for i in range(len(d))]
    val_loss = [d[str(i+1)]["val_loss"] for i in range(len(d))]
    train_acc = [d[str(i+1)]["train_accuracy"] for i in range(len(d))]
    val_acc = [d[str(i+1)]["val_accuracy"] for i in range(len(d))]

    # plot loss/accuracy
    plt.title('monitoring metrics - loss')
    plt.plot(epoch, train_loss, label = "train_loss")
    plt.plot(epoch,val_loss, label = "val_loss")
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(config['res_dir'], 'loss_graph.png'))
    plt.clf()

    plt.title('monitoring metrics - accuracy')
    plt.plot(epoch, train_acc, label = "train_accuracy")
    plt.plot(epoch,val_acc, label = "val_accuracy")
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(config['res_dir'], 'accuracy_graph.png'))
    plt.clf()
    #---------------------------------------------

def main(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    prepare_output(config)

    #mean_std = pkl.load(open(config['dataset_folder'] + '/S2-2017-T31TFM-meanstd.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None

    device = torch.device(config['device'])

    loaders = get_loaders(config)
    for _, (train_loader, val_loader, test_loader) in enumerate(loaders):
        print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))

        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                            dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                            positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                            mlp4=config['mlp4'])

        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)
        else:
            model_config.update(with_extra=False, extra_size=None)

        model = PseTae(**model_config)

        print(model.param_ratio())

        model = model.to(device)
        model.apply(weight_init)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = FocalLoss(config['gamma'])

        trainlog = {}



        best_mIoU = 0
        for epoch in range(1, config['epochs'] + 1):
            print('EPOCH {}/{}'.format(epoch, config['epochs']))

            model.train()
            train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)

            print('Validation . . . ')
            model.eval()
            val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

            print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(val_metrics['val_loss'], val_metrics['val_accuracy'],
                                                                 val_metrics['val_IoU']))

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(trainlog, config)

            if val_metrics['val_IoU'] >= best_mIoU:
                best_mIoU = val_metrics['val_IoU']
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(config['res_dir'], 'model.pth.tar'))

        print('Testing best epoch . . .')
        model.load_state_dict(
            torch.load(os.path.join(config['res_dir'],  'model.pth.tar'))['state_dict'])
        model.eval()

        test_metrics, conf_mat = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                             test_metrics['test_IoU']))
        save_results(test_metrics, conf_mat, config)

    overall_performance(config)
    plot_metrics(config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--dataset_folder', default='', type=str,
                        help='Path to the folder where the results are saved.')

    # set-up data loader folders -----------------------------
    parser.add_argument('--dataset_folder2', default=None, type=str,
                        help='Path to second train folder to concat with first initial loader.')
    parser.add_argument('--val_folder', default=None, type=str,
                        help='Path to the validation folder.')
    parser.add_argument('--test_folder', default=None, type=str,
                        help='Path to the test folder.')

    # ---------------------------add sensor argument to test s1/s2
    parser.add_argument('--sensor', default=None, type=str,
                        help='Type of mission data to train e.g.S1 or S2')

    parser.add_argument('--res_dir', default='./results', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='If specified, the whole dataset is loaded to RAM at initialization')
    parser.set_defaults(preload=False)

    # Training parameters
    parser.add_argument('--kfold', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--npixel', default=64, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[132,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', default=1, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')

    ## TAE
    parser.add_argument('--n_head', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=32, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--mlp3', default='[512,128,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=None, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')

    ## Classifier
    parser.add_argument('--num_classes', default=20, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 20]', type=str, help='Number of neurons in the layers of MLP4')

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k:
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    pprint.pprint(config)
    main(config)
