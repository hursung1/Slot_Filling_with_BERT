import argparse

def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-domain SLU with BERT")
    parser.add_argument("--use_plain", type=bool, default=True, help="if use BERT_NER False elif BERT_Plain True")
    parser.add_argument("--exp_name", type=str, default="Plain_model", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="cross-domain-slu.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")

    # adaptation parameters
    parser.add_argument("--epoch", type=int, default=10, help="number of maximum epoch")
    parser.add_argument("--tgt_dm", type=str, default="", help="target_domain")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_binslot", type=int, default=3, help="number of binary slot O,B,I")
    parser.add_argument("--num_slot", type=int, default=72, help="number of slot types")
    parser.add_argument("--num_domain", type=int, default=7, help="number of domain")
    
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")

    # few shot learning
    parser.add_argument("--n_samples", type=int, default=0, help="number of samples for few shot learning")

    # test model
    parser.add_argument("--model_path", type=str, default="", help="Saved model path")
    parser.add_argument("--model_type", type=str, default="", help="Saved model type (e.g., coach, ct, rzt)")
    parser.add_argument("--test_mode", type=str, default="testset", help="Choose mode to test the model (e.g., testset, seen_unseen)")

    params = parser.parse_args()

    return params
