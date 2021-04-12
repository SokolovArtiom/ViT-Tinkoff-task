import torch
import torch.nn as nn
import numpy as np

import os
import json
import tqdm
from jupyterplot import ProgressPlot
import random

from ViT import VisionTransformer
from Dataset import get_dataset
from Sсheduler import WarmupCosineSchedule

class Trainer:
    def __init__(self, cfg):

        self.cfg = cfg
        self.img_dim = cfg["resolution"]
        self.out_dim = 0
        self._set_seed(self.cfg["seed"])

        os.makedirs(self.cfg["output_directory"], exist_ok=True)

        metrics = ["Train", "Validation"]
            
        plot_names = ["Accuracy"]
                
        print("Metrics to be calculated : {}".format(metrics))
        self.pp = ProgressPlot(
            plot_names,
            line_names=metrics,
            height=350,
            width=750,
            x_label="Epoch",
        )
        self.cur_train_loss = 0
        self.cur_val_loss = 0
        self.glob_step = 0

        if self.cfg["dataset"] == "cifar10":
            self.out_dim=10
        elif self.cfg["dataset"] == "cifar100":
            self.out_dim=100
        elif self.cfg["dataset"] == "flowers":
            self.out_dim = 102
        elif self.cfg["dataset"] == "imagenet":
            self.out_dim=1000

    def train(self):

        model = self._get_model(self.cfg["model_name"]).to(self.cfg["device"])
        optimizer = self._get_optimizer(
            model, self.cfg["optimizer"], self.cfg["lr"], self.cfg["weight_decay"]
        )
        sсheduler = self._get_sсheduler(optimizer, self.cfg["sсheduler"], self.cfg["iters"])
        criterion = self._get_criterion(self.cfg["criterion"])
        train_dl, val_dl = self._get_dataloader()

        os.makedirs(
            os.path.join(
                self.cfg["output_directory"],
                "models/{}/".format(self.cfg["model_name"]),
            ),
            exist_ok=True,
        )

        for epoch in range(1, self.cfg["epochs"] + 1):
            self._train_one_epoch(epoch, model, train_dl, optimizer, criterion, sсheduler)

            print("Epoch : {}, saving the model...".format(epoch))
            torch.save(
                model.state_dict(),
                "{}/models/{}/{}_{}.pth".format(
                    self.cfg["output_directory"],
                    self.cfg["model_name"],
                    self.cfg["unique_name"],
                    epoch,
                ),
            )
            print("Saved!")

            self._validate(epoch, model, val_dl, criterion)
            self.pp.update([[self.cur_train_loss, self.cur_val_loss]])


    def _train_one_epoch(self, epoch, model, train_dl, optimizer, criterion, sсheduler):

        model.train()

        train_loss = 0.0
        all_samples = 0.0
        right_samples = 0.0
        loss = 0.0

        for step, (data, labels) in enumerate(tqdm.tqdm(train_dl)):

            data = data.to(self.cfg["device"])
            labels = torch.squeeze(labels).to(self.cfg["device"])
            predictions = model(data)
            loss = criterion(predictions.view(-1, self.out_dim), labels.view(-1))
            
            loss = loss / self.cfg["acumulate"]
            loss.backward()
            self.glob_step+=1

            if self.glob_step % self.cfg["acumulate"] == 0:

                if self.cfg["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                sсheduler.step()
                model.zero_grad()
                
                with torch.no_grad():
                    train_loss += loss.item()
                    new_right_samples = torch.sum(
                        labels == torch.squeeze(torch.argmax(predictions, axis=1))
                    )
                    right_samples += new_right_samples
                    all_samples += len(labels)

        self.cur_train_loss = right_samples / all_samples
        print("Train accuracy : {}".format(self.cur_train_loss))

    def _validate(self, epoch, model, val_dl, criterion):

        model.eval()

        val_loss = 0.0
        all_samples = 0.0
        right_samples = 0.0

        for step, (data, labels) in enumerate(tqdm.tqdm(val_dl)):
            with torch.no_grad():

                data = data.to(self.cfg["device"])
                labels = torch.squeeze(labels).to(self.cfg["device"])
                predictions = model(data)
                loss = criterion(predictions.view(-1, self.out_dim), labels.view(-1))               
                val_loss += loss.item()

                predictions = predictions.argmax(axis=1)
                right_samples += torch.sum(labels == torch.squeeze(predictions))
                all_samples += len(labels)

        self.cur_val_loss = right_samples / all_samples
        print("Valiadtion accuracy : {}".format(self.cur_val_loss))

    def _get_model(self, model_name):
        
        print("Loading {}...".format(model_name))
        if model_name == "ViT-B_16":
            model =  VisionTransformer(
                        img_dim=self.img_dim,
                        patch_dim=16,
                        out_dim=self.out_dim,
                        num_channels=3,
                        embedding_dim=768,
                        num_heads=12,
                        num_layers=12,
                        hidden_dim=3072,
                        dropout_rate=0.1,
                        warmed_up=self.cfg["warmed_up"],
                        hybrid=False
                    )
            model.zero_grad()
            return model
        elif model_name == "R50+ViT-B_16":
            model =  VisionTransformer(
                        img_dim=self.img_dim,
                        patch_dim=16,
                        out_dim=self.out_dim,
                        num_channels=3,
                        embedding_dim=768,
                        num_heads=12,
                        num_layers=12,
                        hidden_dim=3072,
                        dropout_rate=0.1,
                        warmed_up=self.cfg["warmed_up"],
                        hybrid=True
                    )
            model.zero_grad()
            return model
        else:
            raise ValueError("{} model isn't supported by now".format(model_name))

    def _get_optimizer(self, model, optimizer_name, lr, weight_decay):

        if optimizer_name == "Adam":
            return torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "SGD":
            return torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9
            )
        else:
            raise ValueError(
                "`{}` optimizer isn't supported by now".format(optimizer_name)
            )

    def _get_criterion(self, criterion_name):

        if criterion_name == "CrossEntropyLoss":
            return torch.nn.CrossEntropyLoss()
            raise ValueError("{} isn't supported yet".format(criterion_name))

    def _get_sсheduler(self, optimizer, sсheduler, iters):
        if sсheduler == "Cosine":
            return WarmupCosineSchedule(optimizer, iters/5, iters)

    def _get_dataloader(self):

        return get_dataset(
                    self.cfg["dataset"],
                    self.img_dim,
                    self.cfg["batch_size"],
                    self.cfg["num_workers"]
                )

    def _set_seed(self, seed):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms = True