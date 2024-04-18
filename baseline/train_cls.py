from utils.dataset_classification import *
from utils.augmentation import *
import random
from utils.criterion import ClsCriterion
from torch.utils.data import DataLoader
from utils.metric import Acc
from model import U_Net


def get_model():
    model = U_Net(mode="cls")
    return model


def get_dataloader(batch_size):
    tf_train = Transform2D(p_flip=1, crop=None)
    tf_val = Transform2D(p_flip=0.0, crop=None)

    train_set = DatasetClassificationLow("./dataset/train", tf_train)
    val_set = DatasetClassificationLow("./dataset/val", tf_val)
    len_train = len(train_set)
    len_val = len(val_set)

    dataset_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataset_val = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return dataset_train, len_train,dataset_val, len_val


def save_training_process(train_losses, val_losses, start_val_epoch):
    with open("./process.txt", "w") as file:
        num_epochs = len(train_losses)
        for i in range(num_epochs):
            file.write(f"Epoch {i} Train Loss: {train_losses[i]}")
            if (i + 1) >= start_val_epoch:
                index = i - (start_val_epoch - 1)
                file.write(f"Epoch {i} Val Acc: {val_losses[index]}")
            file.write("\n")


def main():
    # ========================== set random seed ===========================#
    seed_value = 2024  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    # ========================== set hyper parameters =========================#
    BATCH_SIZE = 8
    NUM_EPOCHS = 80
    BASE_LEARNING_RATE = 0.0001
    START_VAL_EPOCH = 40

    # ========================== get model, dataloader, optimizer and so on =========================#
    model = get_model()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LEARNING_RATE,
                                  betas=(0.9, 0.999), weight_decay=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load("./checkpoints/epoch39_val_score_0.885162.pth"))

    dataset_train, len_train, dataset_val, len_val = get_dataloader(BATCH_SIZE)

    criterion = ClsCriterion()  # combined loss function
    evalue = Acc()  # metric to find best model
    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_val_score = 0
    train_losses = []
    val_scores = []

    # ========================== training =============================#
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, sample in enumerate(dataset_train):
            print(f"Epoch[{epoch + 1}/{NUM_EPOCHS}] | Batch {batch_idx}: ", end="")
            batch_train_loss = []
            imgs = sample['image'].to(dtype=torch.float32, device=device)  # (BS, 3, 512, 512)
            labels = sample['label'].to(device=device)

            # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            preds = model(imgs, mode="cls")
            train_loss = criterion(preds, labels)

            # scaler.scale(train_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_train_loss.append(train_loss.detach().cpu().numpy())

            print(f"Batch {batch_idx}:", end="")
        train_losses.append(np.mean(batch_train_loss))

        # ========================= validation =======================#
        if (epoch + 1) >= START_VAL_EPOCH:
            model.eval()
            val_corrects = 0
            with torch.no_grad():
                for batch_idx, sample in enumerate(dataset_val):

                    imgs = sample['image'].to(dtype=torch.float32, device=device)
                    labels = sample['label'].to(device=device)
                    preds = model(imgs, mode="cls")

                    val_correct_cnt = evalue(preds, labels)
                    val_corrects += val_correct_cnt.detach().cpu().numpy()

                val_score = val_corrects / len_val
                val_scores.append(val_score)

                # ================  SAVING ================#
                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), f"./checkpoints/epoch{epoch}_val_acc_{val_score:.6f}.pth")

    save_training_process(train_losses, val_scores, START_VAL_EPOCH)


if __name__ == '__main__':
    main()
