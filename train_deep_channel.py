import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
from torchmetrics import JaccardIndex, Precision, Accuracy, Recall
from tabulate import tabulate

# --- CONFIGURACIÓN GLOBAL ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

BATCH_SIZE = 1 
ACCUMULATION_STEPS = 8 
LEARNING_RATE = 4e-5

NUM_EPOCHS = 100
csv_path = "results_DPT_large_rgbd.csv"

# 
class SargazoDatasetRGBD(Dataset):
    def __init__(self, img_dir, depth_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.mapeo_ids = {16: 1, 21: 2, 23: 3, 24: 4, 30: 5, 101: 6}
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # cargar RGB 
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"[ERROR] No se pudo leer la imagen: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]
        
        # Cargar profundidad y forzar tamaño
        depth_path = os.path.join(self.depth_dir, f"{base_name}_depth.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"[ERROR] Falta mapa de profundidad para: {base_name}")
        depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        depth = np.expand_dims(depth, axis=-1) 
        
        # Concatenar a RGB-D
        rgbd_image = np.concatenate([image, depth], axis=-1) 
        
        # 4. Cargar mascara y forzar tamaño
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
        mask_cruda = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_cruda is None:
            raise ValueError(f"[ERROR] Falta máscara para: {base_name}")
        mask_cruda = cv2.resize(mask_cruda, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        mask_clean = np.zeros_like(mask_cruda, dtype=np.uint8)
        for orig, nuevo in self.mapeo_ids.items():
            mask_clean[mask_cruda == orig] = nuevo
            
        # transformaciones
        if self.transform is not None:
            augmentations = self.transform(image=rgbd_image, mask=mask_clean)
            rgbd_image = augmentations["image"]
            mask_clean = augmentations["mask"]
            
        return rgbd_image, mask_clean

# metricas globales
iou_train = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
prec_train = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(DEVICE)
acc_train = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
rec_train = Recall(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(DEVICE)

iou_val = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
prec_val = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(DEVICE)
acc_val = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
rec_val = Recall(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(DEVICE)

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)
    total_loss = 0
    
    iou_train.reset()
    prec_train.reset()
    acc_train.reset()
    rec_train.reset()

    optimizer.zero_grad()

    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(DEVICE), targets.to(DEVICE).long()

        with torch.amp.autocast(device_type='cuda'):
            predictions = model(data)
            loss = loss_fn(predictions.float(), targets)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 

        with torch.no_grad():
            preds = torch.argmax(predictions.detach(), dim=1)
            iou_train.update(preds, targets)
            prec_train.update(preds, targets)
            acc_train.update(preds, targets)
            rec_train.update(preds, targets)

        total_loss += loss.item() * ACCUMULATION_STEPS
        loop.set_postfix(loss=f"{(loss.item() * ACCUMULATION_STEPS):.4f}")
    
    return total_loss / len(loader)

def valid_fn(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    
    iou_val.reset()
    prec_val.reset()
    acc_val.reset()
    rec_val.reset()

    loop = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for data, targets in loop:
            data, targets = data.to(DEVICE), targets.to(DEVICE).long()

            with torch.amp.autocast(device_type='cuda'):
                predictions = model(data)
                loss = loss_fn(predictions.float(), targets)

            preds = torch.argmax(predictions.detach(), dim=1)
            iou_val.update(preds, targets)
            prec_val.update(preds, targets)
            acc_val.update(preds, targets)
            rec_val.update(preds, targets)

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
    return total_loss / len(loader)

if __name__ == "__main__":
    print(f"Hardware: {DEVICE},  Acumulación: {ACCUMULATION_STEPS})")
    
    # Definición de modelo
    model = smp.DPT(
        encoder_name="tu-vit_large_patch16_384",
        encoder_weights="imagenet",
        in_channels=4,
        classes=NUM_CLASSES,
        #este cambio es solo pra dpt, ya que divide en 384 partes la imagen
        dynamic_img_size=True
    ).to(DEVICE)

    # perdida, optimizador y scaler
    criterion = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device='cuda')

    # Rutas
    train_img_dir = r'C:\Users\sigc\desktop\Sargassum_segmentation\new_training\split_dataset\train_images'
    train_depth_dir = r'C:\Users\sigc\desktop\Sargassum_segmentation\new_training\split_dataset\train_depth'
    train_mask_dir = r'C:\Users\sigc\desktop\Sargassum_segmentation\new_training\split_dataset\train_mask'
    
    val_img_dir = r'C:\Users\sigc\desktop\Sargassum_segmentation\new_training\split_dataset\val_images'
    val_depth_dir = r'C:\Users\sigc\desktop\Sargassum_segmentation\new_training\split_dataset\val_depth'
    val_mask_dir = r'C:\Users\sigc\desktop\Sargassum_segmentation\new_training\split_dataset\val_mask'
    
    # transformaciones a 4 canales
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    #cDataLoaders 
    train_loader = DataLoader(SargazoDatasetRGBD(train_img_dir, train_depth_dir, train_mask_dir, transform=train_transform), 
                              batch_size=BATCH_SIZE, 
                              num_workers= 4, 
                              pin_memory=True,
                              persistent_workers=True,
                              drop_last=True,
                              shuffle=True)

    val_loader = DataLoader(SargazoDatasetRGBD(val_img_dir, val_depth_dir, val_mask_dir, transform=val_transform), 
                            batch_size=BATCH_SIZE, 
                            num_workers=2, 
                            pin_memory=True,
                             persistent_workers=True,
                            drop_last=True, 
                            shuffle=False)

    print(f"\n{'='*75}")
    print(f"Sargasso Segmentation - DPT ")
    print(f"{'='*75}\n")

    history = []
    best_miou = 0.0 
    
    # loop 
    for epoch in range(NUM_EPOCHS):
        avg_train_loss = train_fn(train_loader, model, optimizer, criterion, scaler, epoch)
        avg_val_loss = valid_fn(val_loader, model, criterion)
        
        t_miou = iou_train.compute().item()
        t_prec = prec_train.compute().item()
        t_rec = rec_train.compute().item()
        t_acc = acc_train.compute().item()

        v_miou = iou_val.compute().item()
        v_prec = prec_val.compute().item()
        v_rec = rec_val.compute().item()
        v_acc = acc_val.compute().item()

        history.append([epoch, f"{avg_train_loss:.3f}", f"{avg_val_loss:.3f}", f"{t_miou:.3f}", f"{v_miou:.3f}", f"{t_rec:.3f}", f"{v_rec:.3f}"])
        print("\n" + tabulate(
            history[-10:], 
            headers=['Epoch', 'T-Loss', 'V-Loss', 'T-mIoU', 'V-mIoU', 'T-Rec', 'V-Rec'], 
            tablefmt='grid'
        ))

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'epoch', 'train_loss', 'val_loss', 
                    'train_miou', 'val_miou', 
                    'train_precision', 'val_precision', 
                    'train_recall', 'val_recall', 
                    'train_accuracy', 'val_accuracy'
                ])
            writer.writerow([
                epoch, avg_train_loss, avg_val_loss,
                t_miou, v_miou, t_prec, v_prec,
                t_rec, v_rec, t_acc, v_acc
            ])
        
        # Guardado de modelos
        #torch.save(model.state_dict(), "last_dpt_model.pth")
        
        #if epoch % 10 == 0:
        #    torch.save(model.state_dict(), f"segformer_b5_rgbd_epoch_{epoch}.pth")
            
        if v_miou > best_miou:
            best_miou = v_miou
            torch.save(model.state_dict(), "best_dpt_model.pth")
            print(f"--> [MÁXIMO EN VAL] Época {epoch} | mIoU Val: {best_miou:.4f} | mIoU Train: {t_miou:.4f}")

        
        torch.cuda.empty_cache()