import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import dgl
from dgl.data.utils import load_graphs
from dgl.dataloading import GraphDataLoader
import os
import gc
from dataset import IsoDataset
from config import parse_encoder
from model import Model
import argparse
import logging
import json
import time
import pickle
from utils import adaptive_anchor_ego_net, anchor_ego_net

def train(args, model, optimizer, train_loader, dev_loader, test_loader, device, logger=None):
    best_loss = {"train": float("inf"), "dev": float("inf"), "test": float("inf")}
    best_loss_f1 = {"train": float("inf"), "dev": float("inf"), "test": float("inf")}
    best_epoch = {"train": -1, "dev": -1, "test": -1}
    loss = lambda emb, label: model.margin_loss(emb, label, weight=args.weight)

    for epoch in range(args.epochs):
        model.train()
        num_iter = len(train_loader)
        total_loss = 0
        total_acc, total_prec, total_recall, total_f1, total_auroc = 0, 0, 0, 0, 0
        total_cnt = 1e-6
        t0 = time.time()
        for batch_id, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            q, g, label = batch
            if not args.bidirectional:
                q = dgl.add_reverse_edges(q, copy_ndata=True, copy_edata=True)
                g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
            cnt = label.shape[0]
            total_cnt += cnt
            optimizer.zero_grad()
            q = q.to(device)
            g = g.to(device)
            label = label.to(device)
            pred = model(q, g)
            l = loss(pred, label)
            l.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                e = model.predict(pred).view(-1)
                pred = torch.zeros_like(e, device=e.device)
                pred[e <= config['threshold']] = 1
                total_loss += l
                
            with torch.no_grad():
                pred = pred.cpu()
                label_clssification = label.cpu()

                acc = accuracy_score(label_clssification, pred)
                prec = precision_score(label_clssification, pred, average="binary", zero_division=1)
                recall = recall_score(label_clssification, pred, average="binary", zero_division=1)
                f1 = f1_score(label_clssification, pred, average="binary", zero_division=1)
                total_acc += acc * cnt
                total_prec += prec * cnt
                total_recall += recall * cnt
                total_f1 += f1 * cnt

            if logger and (batch_id % args.print_every == 0 or batch_id == num_iter - 1):
                logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\tmargin loss: {:0>10.3f}\tacc: {:0>3.3f}\tground truth: {:.3f}\tpredict: {:.3f}".format(
                    epoch, args.epochs, "train", batch_id, num_iter, l.item(), acc, label[0].item(), pred[0].item()))
        t1 = time.time()
        mean_time = (t1 - t0) * 1000 / total_cnt
        mean_loss = total_loss / total_cnt
        mean_f1 = total_f1 / total_cnt
        if logger:
            logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tloss: {:0>10.3f}\ttime: {:0>10.3f}".format(
            epoch, args.epochs, "train", mean_loss, mean_time))
            logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tacc: {:0>3.3f}\tprec: {:0>3.3f}\trecall: {:0>3.3f}\tf1: {:0>3.3f}\tauroc: {:0>3.3f}".format(
            epoch, args.epochs, "train", total_acc / total_cnt, total_prec / total_cnt, total_recall / total_cnt,
            total_f1 / total_cnt, total_auroc / total_cnt))

        if(mean_loss < best_loss["train"]):
            best_loss["train"] = mean_loss
            best_loss_f1["train"] = mean_f1
            best_epoch["train"] = epoch
            logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format("train", mean_loss, epoch))
        
        torch.save(model.state_dict(), os.path.join(args.save_model_dir, "epoch%d.pt" % (epoch)))
        dev_loss, dev_f1 = evaluate(args, epoch, model, dev_loader, device, logger, data_type="dev")
        if(dev_loss < best_loss["dev"]):
            best_loss["dev"] = dev_loss
            best_loss_f1["dev"] = dev_f1
            best_epoch["dev"] = epoch
            logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format("dev", dev_loss, epoch))
        test_loss, test_f1 = evaluate(args, epoch, model, test_loader, device, logger, data_type="test")
        if(test_loss < best_loss["test"]):
            best_loss["test"] = test_loss
            best_loss_f1["test"] = test_f1
            best_epoch["test"] = epoch
            logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format("test", test_loss, epoch))
        gc.collect()
    
    for data_type in ["train", "dev", "test"]:
        logger.info("data_type: {:<5s}\tbest mean loss: {:.3f}\tbest mean f1: {:.3f} (epoch: {:0>3d})".format(data_type, best_loss[data_type], best_loss_f1[data_type], best_epoch[data_type]))

def evaluate(args, epoch, model, data_loader, device, logger=None, data_type="valid"):
    num_iter = len(data_loader)
    total_loss = 0
    total_acc, total_prec, total_recall, total_f1, total_auroc = 0, 0, 0, 0, 0
    total_cnt = 1e-6
    
    loss = lambda emb, label: model.margin_loss(emb, label)
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        for batch_id, batch in enumerate(data_loader):
            q, g, label = batch
            cnt = label.shape[0]
            total_cnt += cnt
            if not args.bidirectional:
                q = dgl.add_reverse_edges(q, copy_ndata=True, copy_edata=True)
                g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
            q = q.to(device)
            g = g.to(device)
            label = label.to(device)
            pred = model(q, g)
            l = loss(pred, label)

            e = model.predict(pred).view(-1)
            pred = torch.zeros_like(e, device=e.device)
            pred[e <= config['threshold']] = 1
            total_loss += l
            pred = pred.cpu()
            label_clssification = label.cpu()

            acc = accuracy_score(label_clssification, pred)
            prec = precision_score(label_clssification, pred, average="binary", zero_division=1)
            recall = recall_score(label_clssification, pred, average="binary", zero_division=1)
            f1 = f1_score(label_clssification, pred, average="binary", zero_division=1)
            total_acc += acc * cnt
            total_prec += prec * cnt
            total_recall += recall * cnt
            total_f1 += f1 * cnt

            if logger and (batch_id % args.print_every == 0 or batch_id == num_iter - 1):
                logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\tmargin loss: {:0>10.3f}\tacc: {:0>3.3f}\tground truth: {:.3f}\tpredict: {:.3f}".format(
                    epoch, args.epochs, data_type, batch_id, num_iter, l.item(), acc, label[0].item(), pred[0].item()))
        t1 = time.time()
        mean_time = (t1 - t0) * 1000 / total_cnt
        mean_loss = total_loss / total_cnt
        mean_f1 = total_f1 / total_cnt
        if logger:
            logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tloss: {:0>10.3f}\ttime: {:0>10.3f}".format(
            epoch, args.epochs, data_type, mean_loss, mean_time))
            logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tacc: {:0>3.3f}\tprec: {:0>3.3f}\trecall: {:0>3.3f}\tf1: {:0>3.3f}\tauroc: {:0>3.3f}".format(
            epoch, args.epochs, data_type,total_acc / total_cnt, total_prec / total_cnt, total_recall / total_cnt,
            total_f1 / total_cnt, total_auroc / total_cnt))
        
        return mean_loss, mean_f1

if __name__ == "__main__": 
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parse_encoder(parser)
    args = parser.parse_args()
    
    os.makedirs(args.save_model_dir, exist_ok=True)
    graph, label_dict = load_graphs(os.path.join(args.raw_dir, "graph.dgl"), [0])
    graph = graph[0]
    graph_info = {"max_ngv": graph.number_of_nodes(), 
                  "max_nge": graph.number_of_edges(),
                  "max_ngvl": max(graph.ndata["label"]).item() + 1,
                  "max_ngel": max(graph.edata["label"]).item() + 1}
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config.update(graph_info)
    with open(os.path.join(args.save_model_dir, "train_config.json"), "w") as f:
        json.dump(config, f)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(args.save_model_dir, "train_log.txt"), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info(json.dumps(config, indent=4, ensure_ascii=False))
    device = torch.device("cuda:%d" % args.gpu_id if args.gpu_id != -1 else "cpu")
    if args.gpu_id != -1:
        torch.cuda.set_device(device)
    model = Model(config)
    model = model.to(device)
    logger.info(model)
    logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logger.info('training model')
    collate_fn = None
    if config["resplit"]:
        train_dataset = IsoDataset(raw_dir=os.path.join(args.save_dir, "train"), ego=args.ego, adaptive_id=config['adaptive_id'])
        total_len = len(train_dataset)
        train_len = int(total_len * 8 / 10)
        val_len = int(total_len * 1 / 10)
        test_len = total_len - train_len - val_len
        train_dataset, dev_dataset, test_dataset = random_split(train_dataset, [train_len, val_len, test_len])
    else:
        train_dataset = IsoDataset(raw_dir=os.path.join(args.save_dir, "train"), ego=args.ego, adaptive_id=config['adaptive_id'])
        dev_dataset = IsoDataset(raw_dir=os.path.join(args.save_dir, "dev"), ego=args.ego, adaptive_id=config['adaptive_id'])
        test_dataset = IsoDataset(raw_dir=os.path.join(args.save_dir, "test"), ego=args.ego, adaptive_id=config['adaptive_id'])
    train_loader = GraphDataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dev_loader = GraphDataLoader(dev_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, num_workers=2)
    optimizer = torch.optim.AdamW([param for name, param in model.named_parameters() if name.split('.')[0] != "clf_model"], lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    optimizer.zero_grad()
    train(args, model, optimizer, train_loader, dev_loader, test_loader, device, logger=logger)
