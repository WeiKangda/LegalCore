import os
import sys
import json
from src.model import Model
from src.utils import to_cuda, to_var
from src.utils import get_predicted_clusters, get_event2cluster, fill_expand
from src.metrics import evaluate_documents, b_cubed, ceafe, muc, Evaluator, blanc
import torch
import random
import numpy as np
from tqdm import tqdm
from src.data import get_dataloader, collator
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
import argparse
from torch.optim import Adam
from sklearn.model_selection import KFold
from src.dump_result import dump_result
from pathlib import Path 
from torch.utils.data import DataLoader, Dataset


class EvalResult:
    def __init__(self, gold, mention_to_gold, clusters, mention_to_cluster):
        self.gold = gold
        self.mention_to_gold = mention_to_gold
        self.clusters = clusters
        self.mention_to_cluster = mention_to_cluster

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def evaluate(model, dataloader, metrics, metric_names, desc=""):
    model.eval()
    eval_results = []
    eval_results_lookup = {}
    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])

            probs = model(data)
            for i in range(len(probs)):
                prob = probs[i]
                labels = data["label_groups"][i]
                doc_id = data["doc_id"][i]
                filled_labels = fill_expand(labels)
                filled_labels = to_cuda(filled_labels)

                pred_clusters, pred_event2cluster = get_predicted_clusters(prob)
                gold_event2cluster = get_event2cluster(labels)
                eval_result = EvalResult(labels, gold_event2cluster, pred_clusters, pred_event2cluster)
                eval_results.append(eval_result)
                eval_results_lookup[doc_id] = eval_result.clusters

    print("*"*20 + "Eval Prediction Examples" + "*"*20)
    if len(eval_results) >=5 :
        for i in range(-5, 0):
            print("true:")
            print(eval_results[i].gold)
            print("pred:")
            print(eval_results[i].clusters)
    else:
        for i in range(len(eval_results)):
            print("true:")
            print(eval_results[i].gold)
            print("pred:")
            print(eval_results[i].clusters)

    result_collection = {}
    for metric, name in zip(metrics, metric_names):
        res = evaluate_documents(eval_results, metric)
        result_collection[name] = res
        print(desc + " %s: precision=%.4f, recall=%.4f, f1=%.4f" % (name, *res))
    return result_collection, eval_results, eval_results_lookup

def predict(model, dataloader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predict"):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            probs = model(data)
            for i in range(len(probs)):
                prob = probs[i]
                pred_clusters, pred_event2cluster = get_predicted_clusters(prob)
                all_preds.append({"doc_id": data["doc_id"][i], "clusters": pred_clusters})
    return all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_steps", default=5, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--log_steps", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bert_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--eval_only", action="store_true", help="deprecated")
    parser.add_argument("--dataset", default="maven", type=str, help="[maven, ace, kbp]")
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--sample_rate", default=None, type=float, help="randomly sample a portion of the training data")
    parser.add_argument("--k_folds", default=5, type=int, help="Number of folds for cross-validation")

    args = parser.parse_args()

    output_dir = Path(f"./output/")
    output_dir.mkdir(exist_ok=True, parents=True)
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')
    print(vars(args))

    if args.dataset == "maven":
        get_dataloader = get_dataloader
    else:
        raise NotImplementedError

    set_seed(args.seed)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("loading data...")
    all_data = get_dataloader(tokenizer, "train", data_dir='./data', max_length=args.max_length, shuffle=False, batch_size=args.batch_size, sample_rate=args.sample_rate)

    print("loading model...")
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    metrics = [b_cubed, ceafe, muc, blanc]
    metric_names = ["B-cubed", "CEAF", "MUC", "BLANC"]

    fold_results = []
    all_results = {}
    for fold, (train_indices, val_indices) in enumerate(kfold.split(all_data.dataset)):
        print(f"Starting fold {fold + 1}/{args.k_folds}")

        train_subset = [all_data.dataset[i] for i in train_indices]
        val_subset = [all_data.dataset[i] for i in val_indices]
        print(len(train_subset), len(val_subset))
        train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        dev_dataloader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

        model = Model(len(tokenizer))
        if args.load_ckpt:
            print(f"loading from checkpoint {args.load_ckpt}")
            state_dict = torch.load(args.load_ckpt)["model"]
            model.load_state_dict(state_dict)
        model = to_cuda(model)

        bert_optimizer = AdamW([p for p in model.encoder.model.parameters() if p.requires_grad], lr=args.bert_lr)
        optimizer = Adam([p for p in model.scorer.parameters() if p.requires_grad], lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader) * args.epochs)
        eps = 1e-8

        best_result = {k: (0.0, 0.0, 0.0) for k in metric_names}
        best_eval_results = None
        train_eval_results = []
        train_losses = []
        glb_step = 0

        print("*******************start training********************")
        for epoch in range(args.epochs):
            for data in tqdm(train_dataloader, desc=f"Training epoch {epoch}"):
                model.train()
                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = to_cuda(data[k])
                probs = model(data)
                loss = to_cuda(to_var(torch.tensor(0.0)))
                for i in range(len(probs)):
                    prob = probs[i]
                    labels = data["label_groups"][i]
                    filled_labels = fill_expand(labels)
                    filled_labels = to_cuda(filled_labels)
                    weight = torch.eye(prob.size(0))
                    weight[weight==0.0] = 0.1
                    weight = weight.to(prob.device)
                    prob_sum = torch.sum(torch.clamp(torch.mul(prob, filled_labels), eps, 1-eps), dim=1)
                    loss = loss + torch.sum(torch.log(prob_sum)) * -1
                    pred_clusters, pred_event2cluster = get_predicted_clusters(prob)
                    gold_event2cluster = get_event2cluster(labels)
                    assert len(pred_event2cluster) == len(gold_event2cluster), print(pred_event2cluster, gold_event2cluster)
                    eval_result = EvalResult(labels, gold_event2cluster, pred_clusters, pred_event2cluster)
                    train_eval_results.append(eval_result)

                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                bert_optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                bert_optimizer.zero_grad()
                glb_step += 1

                if glb_step % args.log_steps == 0:
                    print("*"*20 + "Train Prediction Examples" + "*"*20)
                    print("Train %d steps: loss=%f" % (glb_step, np.mean(train_losses)))
                    for metric, name in zip(metrics, metric_names):
                        res = evaluate_documents(train_eval_results, metric)
                        print("Train %d steps %s: precision=%.4f, recall=%.4f, f1=%.4f" % (glb_step, name, *res))
                    train_losses = []
                    train_eval_results = []

                if glb_step % args.eval_steps == 0:
                    res, eval_results, eval_results_lookup = evaluate(model, dev_dataloader, metrics, metric_names, desc="Validation")
                    print(eval_results_lookup)
                    any_better = False
                    for k in res:
                        if k in best_result:
                            if res[k][-1] > best_result[k][-1]:
                                any_better = True
                                best_result[k] = res[k]
                    if any_better:
                        print("better result!")
                        state = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "scheduler": scheduler.state_dict()}
                        torch.save(state, os.path.join(output_dir, f"fold-{fold+1}-best"))
                        best_eval_results = eval_results
                        for k, v in eval_results_lookup.items():
                            all_results[k] = v
        fold_results.extend(best_eval_results)

    print("*"*20 + "Eval Results Across All Folds" + "*"*20)
    result_collection = {}
    for metric, name in zip(metrics, metric_names):
        res = evaluate_documents(fold_results, metric)
        result_collection[name] = res
        print("All folds" + " %s: precision=%.4f, recall=%.4f, f1=%.4f" % (name, *res))
    with open(f"{output_dir}/eval_results.jsonl", "w") as f:
        f.write(json.dumps(all_results) + "\n")
    sys.stdout.close()
