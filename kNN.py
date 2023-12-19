import torch, argparse
from torch_geometric.utils import remove_isolated_nodes
from os.path import join as path_join
import torch.nn.functional as F

from torcheval.metrics import MulticlassAUPRC, MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, BinaryRecall


import graph_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--output_dir", type=str, default="weights/",
                        help="Path to save model weights to")
    parser.add_argument("--path", type=str, default="data/",
                        help="Path to the data folder")
    parser.add_argument("--pt-epoch", type=int, default=950,
                        help="Which epoch to use for the embedder weights")
    parser.add_argument("--k", type=int, default=5,
                        help="The k in kNN")
    parser.add_argument("--combined", action="store_true", default=False,
                        help="Use train and validation dataset")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Use train and validation dataset and test on train")
    args = parser.parse_args()

    # for reproducibility
    graph_utils.set_seed(42)

    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)

    # some paramaters
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    # load graph
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device)
    graph.to(device)

    # removing isolated nodes
    isolated = (remove_isolated_nodes(graph["edge_index"])[2] == False).sum(dim=0).item()
    print(f"Number of isolated nodes = {isolated}\n")

    # define model
    in_channels = graph.num_features
    out_channels = graph.y.shape[1] # number of columns

    embedder_file = f"embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07_{args.pt_epoch}.pt"
    embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(), torch.nn.Linear(in_channels, 128)])
    embedder.load_state_dict(torch.load(path_join(args.output_dir, embedder_file), map_location=device)["state_dict"])
    embedder.to(device)

    train_embeddings = embedder(graph.x[graph.train_idx])
    val_embeddings = embedder(graph.x[graph.val_idx])
    if args.combined:
        embeddings = torch.concat((train_embeddings, val_embeddings))
        test_embeddings = val_embeddings
    elif args.full:
        embeddings = torch.concat((train_embeddings, val_embeddings))
        test_embeddings = embedder(graph.x[graph.test_idx])
    else:
        embeddings = train_embeddings
        test_embeddings = val_embeddings
    
    preds = torch.zeros(len(test_embeddings), dtype=graph.y.dtype, device=graph.y.device)

    for i, emb in enumerate(test_embeddings):
        dist = F.cosine_similarity(emb, embeddings, -1)

        if args.combined:
            # do not want to be neighbours with itself, includes offset
            dist[len(train_embeddings) + i] = -1.

        # get the top k most similar embeddings
        topk = torch.topk(dist, args.k).indices
        labels = graph.y.sum(-1)[topk]

        # majority vote to get prediction
        preds[i] = torch.mode(labels, 0).values
    
    # Evaluation metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    macro_precision = MulticlassPrecision(num_classes=4, average="macro")
    binary_recall = BinaryRecall()
    macro_AUPRC = MulticlassAUPRC(num_classes=4, average="macro")

    if args.full:
        y = torch.sum(graph.y[graph.test_idx], dim=-1).long()
    else:
        y = torch.sum(graph.y[graph.val_idx], dim=-1).long()

    # Train and valuation confusion matrices
    conf_mat = graph_utils.confusion_matrix(conf, preds, y)

    # Train and valuation accuracy
    accuracy = graph_utils.accuracy(acc, preds, y)

    # Train and valuation macro recall
    m_recall = graph_utils.macro_recall(macro_recall, preds, y)

    # Train and valuation macro recall
    m_precision = graph_utils.macro_precision(macro_precision, preds, y)

    # Train and valuation binary accuracy
    binary_mask = torch.logical_or((y == 0), (y == 3))
    y_binary = graph_utils.rewrite_labels_binary(y[binary_mask])
    y_binary_pred = graph_utils.rewrite_labels_binary(preds[binary_mask])
    b_recall = graph_utils.binary_recall(binary_recall, y_binary_pred, y_binary)

    # One frame agreement
    ofa = graph_utils.k_frame_agreement(preds, y, k=1)

    # Valuation macro area under the precision-recall curve
    m_AUPRC = graph_utils.macro_AUPRC(macro_AUPRC, preds, y)

    # Print train and valuation confusion matrices
    print(f"Confusion matrix:\n\t{conf_mat.long()}")
    # Print valuation accuracy
    print(f"Accuracy: {accuracy.item()}")
    # Print valuation macro recall
    print(f"Macro recall: {m_recall.item()}")
    # Print valuation macro recall
    print(f"Macro precision: {m_precision.item()}")
    # Print valuation binary accuracy
    print(f"Binary recall: {b_recall.item()}")
    # Print valuation one frame agreement
    print(f"One frame agreement: {ofa}")
    # Print valuation macro AUPRC
    print(f"Macro AUPRC: {m_AUPRC.item()}")
 