from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics.classification import MulticlassAccuracy
import torch

DEVICE = 'cuda'
NUM_CLASSES = 10

def train(model, optimizer, loader, criterion, criterion_addition = lambda x, y: 0, criterion_addition_name: str = "0"): # possible: add addition
    '''
        criterion(logits, labels)

        logits: [batch_size, K] float tensor
        labels: [batch_size] int tensotr
    '''
    model.train()
    losses_tr = []
    losses_tr_additions = []

    calibration_metric = MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=15).to(DEVICE)
    accuracy_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average='micro').to(DEVICE)

    for images, labels in tqdm(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss_addition = criterion_addition(logits, labels)

        loss += loss_addition
        loss.backward()
        optimizer.step()

        losses_tr.append(loss.item())
        losses_tr_additions.append(float(loss_addition))

        calibration_metric(logits, labels)
        accuracy_metric(logits, labels)

    return model, optimizer, {
        'full_loss': np.mean(losses_tr),
        f'addition_{criterion_addition_name}_loss': np.mean(losses_tr_additions),
        'acc_micro_1': accuracy_metric.compute().item(),
        'calib': calibration_metric.compute().item()
    }

def val(model, loader, criterion, criterion_addition = lambda x, y: 0, criterion_addition_name: str = "0"):
    model.eval()
    losses_val = []
    losses_val_addition = []

    calibration_metric = MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=15).to(DEVICE)
    accuracy_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average='micro').to(DEVICE)

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)

            loss = criterion(logits, labels)
            loss_addition = criterion_addition(logits, labels)
            loss += loss_addition

            losses_val.append(loss.item())
            losses_val_addition.append(float(loss_addition))

            calibration_metric(logits, labels)
            accuracy_metric(logits, labels)


    return {
        'full_loss': np.mean(losses_val),
        f'addition_{criterion_addition_name}_loss': np.mean(losses_val_addition),
        'acc_micro_1': accuracy_metric.compute().item(),
        'calib': calibration_metric.compute().item()
    }



def create_model_and_optimizer(model_class, model_params, lr=1e-3, beta1=0.9, beta2=0.999, device=DEVICE):
    model = model_class(**model_params)
    model = model.to(device)

    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

    optimizer = torch.optim.Adam(params, lr, [beta1, beta2])
#     optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    return model, optimizer




from IPython.display import clear_output
import warnings
import os
from collections import defaultdict
import matplotlib.pyplot as plt


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def learning_loop(
    model,
    optimizer,
    train_loader,
    val_loader,
    criterion,
    criterion_addition = lambda x, y: 0,
    criterion_addition_name: str = "0",
    scheduler=None,
    min_lr=None,
    epochs=10,
    val_every=1,
    draw_every=1,
    separate_show=False,
    model_name=None,
    chkp_folder="./chkps",
    metric_names=None,
    save_only_best=True,
    run=None, # neptune logger
):
    if model_name is None:
        if os.path.exists(chkp_folder):
            num_starts = len(os.listdir(chkp_folder)) + 1
        else:
            num_starts = 1
        model_name = f'model#{num_starts}'

    if os.path.exists(os.path.join(chkp_folder, model_name)):
        model_name = model_name + "_v2"
        warnings.warn(f"Selected model_name was used already! To avoid possible overwrite - model_name changed to {model_name}")
    os.makedirs(os.path.join(chkp_folder, model_name))

    losses = {'train': [], 'val': []}
    lrs = []
    best_val_loss = np.Inf

    metrics = defaultdict(list)

    for epoch in range(1, epochs+1):
        print(f'#{epoch}/{epochs}:')

        lrs.append(get_lr(optimizer))

        model, optimizer, train_metrics = train(model, optimizer, train_loader, criterion, criterion_addition, criterion_addition_name)
        # train_metrics: dict with keys: 'loss', 'acc_micro_1', 'calib'
        losses['train'].append(train_metrics['full_loss'])

        if run is not None:
            for metr_name, metr_val in train_metrics.items():
                run[f'train/{metr_name}'].append(metr_val)

        if not (epoch % val_every):
            val_metrics = val(model, val_loader, criterion, criterion_addition, criterion_addition_name)
            loss = val_metrics['full_loss']
            losses['val'].append(loss)
            if val_metrics is not None:
                for metr_name, metr_val in val_metrics.items():
                    # metrics[metr_name].append(metr_val)

                    if run is not None:
                        run[f'val/{metr_name}'].append(metr_val)

            # Сохраняем лучшую по валидации модель
            if ((not save_only_best) or (loss < best_val_loss)):
                if not os.path.exists(chkp_folder):
                    os.makedirs(chkp_folder)

                run.wait() # to allow all results to be uploaded
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'losses': losses,
                        'train_metrics': {metr_name: run[f'train/{metr_name}'].fetch_values()['value'].to_list() for metr_name in metric_names},
                        'val_metrics': {metr_name: run[f'val/{metr_name}'].fetch_values()['value'].to_list() for metr_name in metric_names},
                    },
                    os.path.join(chkp_folder, model_name, f'{model_name}#{epoch}.pt'),
                )
                best_val_loss = loss

            if scheduler:
                try:
                    scheduler.step()
                except:
                    scheduler.step(loss)


        ###############

        if not (epoch % draw_every):
            clear_output(True)
            ww = 2 if metric_names else 1
            fig, ax = plt.subplots(1, ww, figsize=(20, 10))
            fig.suptitle(f'#{epoch}/{epochs}:')

            plt.subplot(1, ww, 1)
            plt.title('losses')
            plt.plot(losses['train'], 'r.-', label='train full')
            plt.plot(losses['val'], 'g.-', label='val full')
            plt.legend()

            if metric_names:
                plt.subplot(1, ww, 2)
                plt.title('additional metrics')

                run.wait() # to allow all results upload
                for metr_name in metric_names:
                    metric_train_values = run[f'train/{metr_name}'].fetch_values()['value'].to_list()
                    metric_val_values = run[f'val/{metr_name}'].fetch_values()['value'].to_list()
                    plt.plot(metric_train_values, '.-', label=f'train/{metr_name}')
                    plt.plot(metric_val_values, '.-', label=f'val/{metr_name}')
                plt.legend()

            plt.show()

        if min_lr and get_lr(optimizer) <= min_lr:
            print(f'Learning process ended with early stop after epoch {epoch}')
            break

    return model, optimizer, losses




def learning_loop_double(
    model,
    model_2,
    optimizer,
    optimizer_2,
    train_loader,
    val_loader,
    criterion,
    criterion_addition = lambda x, y: 0,
    criterion_addition_name: str = "0",
    scheduler=None,
    min_lr=None,
    epochs=10,
    val_every=1,
    draw_every=1,
    separate_show=False,
    model_name=None,
    chkp_folder="./chkps",
    metric_names=None,
    save_only_best=True,
    run=None, # neptune logger
):
    '''
        model with addition
        model_2 with NO addition
    '''
    if model_name is None:
        if os.path.exists(chkp_folder):
            num_starts = len(os.listdir(chkp_folder)) + 1
        else:
            num_starts = 1
        model_name = f'model#{num_starts}'

    if os.path.exists(os.path.join(chkp_folder, model_name)):
        model_name = model_name + "_v2"
        warnings.warn(f"Selected model_name was used already! To avoid possible overwrite - model_name changed to {model_name}")
    os.makedirs(os.path.join(chkp_folder, model_name))

    losses = {'train': [], 'val': []}
    lrs = []
    best_val_loss = np.Inf

    metrics = defaultdict(list)

    for epoch in range(1, epochs+1):

        ### model 1 loop
        print(f'#{epoch}/{epochs}:')

        lrs.append(get_lr(optimizer))

        model, optimizer, train_metrics = train(model, optimizer, train_loader, criterion, criterion_addition, criterion_addition_name)
        model_2, optimizer_2, train_metrics_2 = train(model_2, optimizer_2, train_loader, criterion)
        # train_metrics: dict with keys: 'loss', 'acc_micro_1', 'calib'
        losses['train'].append(train_metrics['full_loss'])

        if run is not None:
            for metr_name, metr_val in train_metrics.items():
                run[f'train/{metr_name}'].append(metr_val)

            for metr_name, metr_val in train_metrics_2.items():
                run[f'train_no_add/{metr_name}'].append(metr_val)

        if not (epoch % val_every):
            val_metrics = val(model, val_loader, criterion, criterion_addition, criterion_addition_name)
            val_metrics_2 = val(model_2, val_loader, criterion)

            loss = val_metrics['full_loss']
            losses['val'].append(loss)
            if val_metrics is not None:
                for metr_name, metr_val in val_metrics.items():
                    # metrics[metr_name].append(metr_val)

                    if run is not None:
                        run[f'val/{metr_name}'].append(metr_val)

                for metr_name, metr_val in val_metrics_2.items():
                    # metrics[metr_name].append(metr_val)

                    if run is not None:
                        run[f'val_no_add/{metr_name}'].append(metr_val)

            # Сохраняем лучшую по валидации модель
            if ((not save_only_best) or (loss < best_val_loss)):
                if not os.path.exists(chkp_folder):
                    os.makedirs(chkp_folder)

                run.wait() # to allow all results to be uploaded
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'losses': losses,
                        'train_metrics': {metr_name: run[f'train/{metr_name}'].fetch_values()['value'].to_list() for metr_name in metric_names},
                        'val_metrics': {metr_name: run[f'val/{metr_name}'].fetch_values()['value'].to_list() for metr_name in metric_names},
                        'train_no_add_metrics': {metr_name: run[f'train_no_add/{metr_name}'].fetch_values()['value'].to_list() for metr_name in metric_names[:-1]},
                        'val_no_add_metrics': {metr_name: run[f'val_no_add/{metr_name}'].fetch_values()['value'].to_list() for metr_name in metric_names[:-1]},

                    },
                    os.path.join(chkp_folder, model_name, f'{model_name}#{epoch}.pt'),
                )
                best_val_loss = loss

            if scheduler:
                try:
                    scheduler.step()
                except:
                    scheduler.step(loss)


        ###############

        if not (epoch % draw_every):
            clear_output(True)
            ww = 2 if metric_names else 1
            fig, ax = plt.subplots(1, ww, figsize=(20, 10))
            fig.suptitle(f'#{epoch}/{epochs}:')

            plt.subplot(1, ww, 1)
            plt.title('losses')
            plt.plot(losses['train'], 'r.-', label='train full')
            plt.plot(losses['val'], 'g.-', label='val full')
            plt.legend()

            if metric_names:
                plt.subplot(1, ww, 2)
                plt.title('additional metrics')

                run.wait() # to allow all results upload
                for metr_name in metric_names:
                    metric_train_values = run[f'train/{metr_name}'].fetch_values()['value'].to_list()
                    metric_val_values = run[f'val/{metr_name}'].fetch_values()['value'].to_list()
                    plt.plot(metric_train_values, '.-', label=f'train/{metr_name}')
                    plt.plot(metric_val_values, '.-', label=f'val/{metr_name}')
                plt.legend()

            plt.show()

        if min_lr and get_lr(optimizer) <= min_lr:
            print(f'Learning process ended with early stop after epoch {epoch}')
            break

    return model, optimizer, losses
