import torch
import torch.nn as nn
import torchvision
import os


class ImageNetVal(object):
    def __init__(self, model, model_type, device):
        self.name = 'val'
        self.model = model
        self.model_type = model_type
        self.device = device
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(self.device)
        self.imagenet_val_path = "imagenet_val"

    def data(self):
        if model_type in ["vone-alexnet",
                          "vone-resnet50",
                          "vone-resnet50_at",
                          "vone-resnet50_ns",
                          "vone-cornets"]:
            norm_transform = torchvision.transforms.Normalize(
                                             mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])
        else:
            norm_transform = torchvision.transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        dataset = torchvision.datasets.ImageFolder(
            self.imagenet_val_path,
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalization_transform,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=128,
                                                  shuffle=False,
                                                  num_workers=20,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                target = target.to(self.device)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)
        print(record)
        return record


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == "__main__":
    all_models = [
                  'alexnet',
                  'vgg',
                  'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  'efficientnet_b3', 'resnext50_32x4d',
                  "vone-alexnet",
                  "vone-resnet50",
                  "vone-resnet50_at",
                  "vone-resnet50_ns",
                  "vone-cornets"
                  ]

    model_type = "vone-resnet50"

    if model_type == "alexnet":
        model = load_alexnet()
    elif model_type in ['resnet18', 'resnet34', 'resnet50',
                        'resnet101', 'resnet152']:
        model = load_resnet(model_type)
    elif model_type == "vgg":
        model = load_vgg()
    elif model_type in ["vone-alexnet",
                        "vone-resnet50",
                        "vone-resnet50_at",
                        "vone-resnet50_ns",
                        "vone-cornets"]:
        model_name = model_type.split("-")[1]
        model = load_vonenet(model_name)
    else:
        model = load_timm(model_type)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    validator = ImageNetVal(model, model_type, device)
    record = validator()
    print(record['top1'])
    print(record['top5'])