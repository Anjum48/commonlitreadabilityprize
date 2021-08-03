
I'd like to thank Kaggle and CommonLit for hosting this competition, as well has everyone here who has shared such great discussions and notebooks. I'm still in shock as I've always dreamed of a solo gold but never thought I'd actually get here!

# CV Strategy
This was my second NLP competitions after the [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction) competition last year. There was some similarity between the two, namely a) a small dataset and b) a fairly noisy target. The trick with the TSE competition was to use a 5-fold CV but to average multiple seeds (I used 5 for a total of 25 runs per experiment, which I would then average). 

To create my 5 folds, I used [Abhishek's binning and stratification](https://www.kaggle.com/abhishek/step-1-create-folds) method. On average I was seeing about 0.01 gap between CV & public LB, but the correlation was good. I didn't bother submitting the majority of my single models as I was happy with my CV setup and I'm lazy 😂️.

My best performing model was a deberta-large with a CV of 0.4738 (average of 5 seeds).

# Pipeline
## Transformers
For all of my transformer models I used the [`AttentionBlock` from this notebook](https://www.kaggle.com/gogo827jz/roberta-model-parallel-fold-training-on-tpu), but applied it to the to the sequence output of the transformer. The idea behind this was to get the the transformer to attend to the full sentence in a more learnable way than max or mean pooling.

```
n_hidden = self.config.hidden_size  # from HF config

self.seq_attn_head = nn.Sequential(
    nn.LayerNorm(n_hidden),
    AttentionBlock(n_hidden, n_hidden, 1),
    nn.Linear(n_hidden + n_additional_features, 2 if kl_loss else 1),
)
```

## Additional features
I also added a couple of features using `textstat`, `flesch_reading_ease` and `smog_index` and concatenated them to the output of the AttentionBlock before the regression head. These brought a tiny but repeatable improvement to my CV, but I'm not sure if this had any major impact on the final performance. I did a test during the late stages of the competition where I concatenated predictions of other models, and this didn't work any better than the ensemble methods below, which makes me think these features were doing very little.

## Training config
* Optimiser: AdamW
    * Weight decay: 1.0
    * LR: 0.00005 or 0.000025 (depending on batch size)
* Epochs: 6 with SWA after epoch 3
* Batch size: 16 (or 12 for the larger models)
* Validation/Checkpointing every 5 steps

## Other tools
* Logging: combination of Weights & Biases and Google Sheets (to track multiple seeds)
* Model Store: Google Cloud Bucket and transfer to a Kaggle Notebook (to /kaggle/working). I found this was much more reliable than regular dataset uploads, plus you won't run into storage limitations.
* PyTorch Lightning
* Shell script using Kaggle API to measure notebook execution times

# Ensemble
## Selection
I used 3 main methods for ensembling:
* `RidgeCV` (which uses `LeaveOneOut` cross validation)
* `BayesianRidgeRegression` (with a LOO CV loop)
* The [Netflix method from 2009 BigChaos solution](https://kaggler.readthedocs.io/en/latest/_modules/kaggler/ensemble/linear.html#netflix) ([paper](https://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf)). This method uses RMSE scores from CV to allocate ensemble weights

I started with all of my models and iteratively removed the worst models until I could get down to the GPU time limit. (I found that this worked better than greedily adding models).

I think this worked well in the end since the selection process would select models that will help the overall ensemble instead of models with the best CV.

## Final submission
My final ensembles usually had about 38 models to fit into the 3 hour run time and had quite a range of model architectures and CV scores.

I chose the RidgeCV & BayesianRidgeRegression methods as my final 2 submissions with the former scoring 0.447. 

Interestingly, the Netflix method scored 0.446, but I had more faith in the CV scores from LOO CV, as I was worried about leakage using the Netflix method.


| Model                                          |       CV |
|:-----------------------------------------------|---------:|
| microsoft/deberta-large                        | 0.484005 |
| microsoft/deberta-base                         | 0.486186 |
| deepset/roberta-base-squad2                    | 0.495078 |
| deepset/roberta-base-squad2                    | 0.501851 |
| deepset/roberta-large-squad2                   | 0.482696 |
| distilroberta-base                             | 0.505654 |
| funnel-transformer/large-base                  | 0.495666 |
| bert-large-uncased                             | 0.524264 |
| bert-large-uncased                             | 0.538173 |
| facebook/bart-base                             | 0.534491 |
| facebook/bart-base                             | 0.556662 |
| facebook/bart-large                            | 0.543126 |
| facebook/bart-large                            | 0.526896 |
| roberta-base                                   | 0.49835  |
| microsoft/deberta-large                        | 0.474008 |
| microsoft/deberta-large                        | 0.471707 |
| distilroberta-base                             | 0.502802 |
| albert-large-v2                                | 0.545104 |
| albert-large-v2                                | 0.506411 |
| funnel-transformer/large-base                  | 0.493789 |
| funnel-transformer/large-base                  | 0.491098 |
| microsoft/deberta-base                         | 0.499093 |
| microsoft/deberta-base                         | 0.509152 |
| deepset/roberta-base-squad2                    | 0.490829 |
| deepset/roberta-large-squad2                   | 0.489583 |
| deepset/roberta-large-squad2                   | 0.494185 |
| funnel-transformer/large-base                  | 0.523333 |
| albert-large-v2                                | 0.508817 |
| albert-large-v2                                | 0.529876 |
| sentence-transformers/LaBSE                    | 0.525731 |
| microsoft/deberta-large                        | 0.471197 |
| microsoft/deberta-large                        | 0.475243 |
| bert-large-cased-whole-word-masking            | 0.514011 |
| bert-large-cased                               | 0.505559 |
| xlm-roberta-large                              | 0.505576 |
| facebook/bart-base                             | 0.530183 |
| google/electra-large-discriminator             | 0.514033 |
| sentence-transformers/paraphrase-mpnet-base-v2 | 0.510096 |
