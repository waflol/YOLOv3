"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

from config import yolov3_cfg as configs
import torch
import torch.optim as optim

from models.models import YOLOv3
from tqdm import tqdm
from utils.torch_utils import *
from utils.losses import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(configs.DEVICE)
        y0, y1, y2 = (
            y[0].to(configs.DEVICE),
            y[1].to(configs.DEVICE),
            y[2].to(configs.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    model = YOLOv3(num_classes=configs.NUM_CLASSES).to(configs.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=configs.LEARNING_RATE, weight_decay=configs.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=configs.DATASET + "/train.csv", test_csv_path=configs.DATASET + "/test.csv"
    )

    if configs.LOAD_MODEL:
        load_checkpoint(
            configs.CHECKPOINT_FILE, model, optimizer, configs.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(configs.ANCHORS)
        * torch.tensor(configs.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(configs.DEVICE)

    for epoch in range(configs.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=configs.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=configs.NMS_IOU_THRESH,
                anchors=configs.ANCHORS,
                threshold=configs.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=configs.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=configs.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    main()
