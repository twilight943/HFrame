import argparse

def add_args(parser):
    parser.add_argument("--base", type=int,
            help="encoding base")
    
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--print_every", type=int,
            help="log info each print_every iteration")
    
    parser.add_argument("--resplit", action="store_true")
    parser.add_argument("--ego", action="store_true")
    parser.add_argument("--adaptive_id", action="store_true")
    parser.add_argument("--rglr_mlpid", action="store_true")
    parser.add_argument("--homo", action="store_true")
    parser.add_argument("--init_emb", type=str)

    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_grad_norm", type=int)

    parser.add_argument("--layer_type", type=str)
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--emb_out_dim", type=int)

    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_mlp_layers", type=int)
    parser.add_argument("--net_hidden_dim", type=int)
    parser.add_argument("--layer_aggr_type", type=str)
    parser.add_argument("--batch_norm", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--init_eps", type=float)
    parser.add_argument("--rand_eps", action="store_true")
    parser.add_argument("--learn_eps", action="store_true")

    parser.add_argument("--conv", type=str)
    parser.add_argument("--rgin_num_bases", type=int)
    parser.add_argument("--rgin_regularizer", type=str)
    parser.add_argument("--emb_hidden_dim", type=int)
    parser.add_argument("--predict_hidden_dim", type=int)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--add_feat_enc", action="store_true")
    parser.add_argument("--add_id_enc", action="store_true")
    parser.add_argument("--id_enc_dim", type=int)
    parser.add_argument("--add_pos_enc", action="store_true")
    parser.add_argument("--order_emb", action="store_true")
    parser.add_argument("--skip_layer", action="store_true")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--aggr_norm", action="store_true")
    parser.add_argument("--nonneg", action="store_true")

    parser.add_argument("--margin", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--weight", type=float, help='positive sample weight in loss')

    parser.add_argument("--dualsim", action="store_true")
    parser.add_argument("--embed", action="store_true")

    parser.add_argument("--raw_dir", type=str,
            help="raw graph dir")
    parser.add_argument("--save_dir", type=str,
            help="sample data dir")
    parser.add_argument("--save_model_dir", type=str,
            help="save model to save_model_dir each epoch")

    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--n_workers", type=int)


def parse_encoder(parser):
    add_args(parser)
    parser.set_defaults(
        base=2,
        gpu_id=0,
        num_workers=48,
        epochs=100,
        batch_size=32,
        print_every=500,

        init_emb="Equivariant",
        add_id_enc=False,
        id_enc_dim=3,
        homo=True,
        ego=True,
        adaptive_id=True,
        rglr_mlpid=True,

        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-5,
        max_grad_norm=0.0,

        add_feat_enc=True,
        bidirectional=True,
        layer_type="hgin",
        emb_dim=64,
        emb_out_dim=64,
        
        num_layers=4,
        num_mlp_layers=3,
        net_hidden_dim=64,
        
        layer_aggr_type="sum",
        batch_norm=True,
        residual=True,
        init_eps=1e-3,
        learn_eps=True,

        abs=False,
        aggr_norm=False,
        nonneg=False,
        
        margin=1.0,
        threshold=0.5,

        filter_net="None",
        conv="DisRGIN",
        rgin_num_bases=2,
        rgin_regularizer="bdd",
        emb_hidden_dim=64,
        predict_hidden_dim=64,
        weight=1,

        raw_dir="../datasets/cora",
        save_dir="../datasets/cora/sample",
        save_model_dir="../dumps/cora",
        n_workers=48,
        layer_num=1,
        anchor_num=32,
        anchor_size_num=4)
