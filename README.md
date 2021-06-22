# Cross Domain Slot Filling with BERT

* Data: SNIPS Dataset preprocessed by [zliucr](https://github.com/zliucr/coach)
* Reference: Download data from [here](https://drive.google.com/drive/folders/1ydalMtB-hpfS3SIEaR5UbRfEe2m8bFcj)

## How to train
* run <b>run.sh</b>

## How to test (evaluation)
* run <b>run_test_total.sh</b> or <b>run_test_seen_unseen.sh</b>
    * Test dataset contains data which includes slot labels <b>not in source domain</b>. These data are called <i>unseen data</i>. Otherwise, the data are called <i>seen data</i>.
    * run_test_total.sh: evaluate on test datasets <b>w/o data split</b>.
    * run_test_seen_unseen.sh: <b>split test dataset into to sets</b>(seen dataset, unseen dataset) then evaluate on each datasets.

## Options
* --epoch: training epochs
* --tgt_dm: target domain, source domain consists of domains without except target domain
* --batch_size: batch size
* --lr: learning rate
* --dropout: dropout rate
* --n_samples: Used in training step. The number of samples for <i>few shot learning</i>. <b>Include n_samples number of target domain in source domain</b>.
* --test_mode: Used in test step. Set test mode as "<b>testset</b>" for <i>run_test_total.sh</i> or "<b>seen_unseen</b>" for <i>run_test_seen_unseen.sh</i>