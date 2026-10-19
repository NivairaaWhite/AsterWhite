# Model Card -- AsterWhite-small

A small, genuinely trained AsterWhite language model, included so that
`generate` and `serve` work immediately after unzipping. It is a
**demonstration-scale** model (6.5M parameters, ~1.5M training tokens), not a
frontier or production-quality writer. Read "What it can and cannot do" before
building anything on it.

## Summary
- **Architecture**: AsterWhite decoder, attention-only (`layer_pattern: "A"`), 6 layers,
  d_model 288, 6 query heads / 2 KV heads (grouped-query attention, RoPE), SwiGLU
  d_ff 768, RMSNorm, tied input/output embeddings. No MoE, no multi-token prediction.
- **Parameters**: 6,491,808 (26 MB on disk)
- **Context length**: 512 tokens (`inference.max_total_tokens` is capped there; longer
  generations use the sliding-window KV trim, they do not extend what the model learned)
- **Tokenizer**: byte-level BPE, 4,096 tokens (incl. AsterWhite control tokens), trained on
  the same corpus. SHA-256 `daaee7e9305e952d6d27effa17223277d239498f18d1886055a4f948a65c2f70`
  (checked on every load; a mismatching tokenizer is refused).
- **Weights shipped**: EMA (Polyak-averaged, decay 0.998) of the training run at step 2250 of
  3000 -- the point of lowest held-out perplexity (see Evaluation).
- **Intended use**: trying out the AsterWhite pipeline (generate / serve / align / fine-tune),
  tests, demos, a starting point for continued training or LoRA fine-tuning.
- **Out-of-scope use**: factual Q&A, code, advice of any kind, safety-critical use, and any
  public-facing product without your own evaluation and safety work. The model has **no
  instruction tuning and no safety tuning**; it only continues text.

## Training data
Public-domain English fiction from Project Gutenberg, via the NLTK `gutenberg` corpus,
rebuilt reproducibly by `scripts/prepare_corpus.py`:

| Author | Works |
|---|---|
| Jane Austen | *Emma*, *Persuasion*, *Sense and Sensibility* |
| Herman Melville | *Moby Dick* |
| G. K. Chesterton | *The Ball and the Cross*, *The Man Who Was Thursday*, *The Wisdom of Father Brown* |
| Lewis Carroll | *Alice's Adventures in Wonderland* |
| Maria Edgeworth | *The Parent's Assistant* |
| Sara Cone Bryant | *Stories to Tell to Children* |
| Thornton W. Burgess | *The Adventures of Buster Bear* |

- 5,554,582 characters of training text (1,519,525 tokens); 227,252 characters (~63k tokens)
  held out for validation.
- Poetry, plays and scripture were deliberately excluded; Gutenberg's hard-wrapped lines were
  un-wrapped into paragraphs so the model does not learn to break lines every 70 columns.
- Everything is 19th-/early-20th-century English, so that is the only register the model has.
  Modern vocabulary, dialogue style, names and non-English text are essentially absent.

## Training
- Recipe: `configs/pretrained_small.yaml` (reproduce with the commands in the README).
- Optimizer: AdamW, peak LR 1e-3, 150 warm-up steps, cosine decay, weight decay 0.1, fp32,
  dropout 0, batch 8 x 512 tokens (4,096 tokens/step), 3,000 steps (~12.3M tokens processed,
  about 8 passes over the training text). `--seed` makes runs reproducible.
- Hardware: 2 CPU cores, roughly 3,900 tokens/s -- about an hour of compute. The sandbox
  restarted the process twice; the run was resumed from its own checkpoints (weights, optimizer,
  RNG, step count restored).
- No alignment (GRPO), no LoRA, no instruction data.

## Evaluation
Held-out next-token perplexity on the validation split (512-token blocks; reproduce with
`python train.py evaluate --data <val_dir>`, which uses this bundle by default):

| Model | Perplexity |
|---|---|
| Uniform guess over 4,096 tokens | 4,096 |
| Unigram (word-frequency) baseline | ~787 |
| Raw weights, step 250 / 1000 / 2250 / 3000 | 166 / 68 / 55.3 / 56.4 |
| **Shipped: EMA weights, step 2250** | **52.8** (val loss 3.9665) |

Validation perplexity bottomed out around step 2250 and then rose slightly while training loss
kept falling (about 3.1 at the end vs. ~4.0 validation): the model begins to memorise a
1.5M-token corpus, which is why the best checkpoint, not the last one, ships.

**How to read that number honestly**
- The validation text is the *last 4% of every book*, so the same books, characters and
  vocabulary appear in training. This measures "continue a held-out passage of a book it has seen",
  not generalisation to new authors; on a different book expect noticeably worse perplexity.
- The checkpoint was chosen by that same validation set, which flatters it slightly.
- The unigram figure is computed on whole documents rather than 512-token blocks, so it is an
  approximate reference point, not an exact like-for-like number.

### Sample outputs (seeded, default sampling: temperature 0.8, top-p 0.92, top-k 50)
Prompt `It was a dark and stormy night,` (`--seed 11`):

> and with sunlight fidgeing, burnished gold, broken in lamps, and the collar light borders and
> aliance. For one morning, through which was a rout and broken head was still on him, with a host
> of a splinter to the knelt of his dry. ...

Prompt `"I do not think,`:

> though there was no very older in his state; and he had no friend with us, and he was not so
> plain as to be doubting, that he was fitted to look for his wife, that she did not think the
> Cox must give him a piece of his hand ...

Prompt `Chapter One.` at temperature 0.6 (more fluent, but loops):

> The Frog of Mountain is not a man, but he is a very good-natured man, and he is a man. He is
> a good-natured man. But there is no other thing in the world. ...

### What it can and cannot do
- **Can**: produce text with plausible spelling, punctuation, paragraphing, quoted dialogue with
  "said ...", and a recognisably 19th-century cadence; run the full AsterWhite pipeline end to end.
- **Cannot**: stay on a topic for more than a sentence or two, keep characters or plot straight,
  follow instructions, or tell the truth about anything. It invents words ("fidgeing", "aliance")
  and names, mixes characters from different books (a whale, a Frog, a Cox), and repeats itself at
  low temperature.
- **Sampling**: temperature 0.6-0.7 reads more fluently but repeats; 0.8-1.0 is more inventive and
  less coherent. There is no setting that makes it coherent over a chapter -- that is a
  model-size and data limit.

### The story planner is not trained
`GenerationOrchestrator`'s planner, universe tracker and character encoder are randomly
initialised modules. This model was pretrained as a plain language model, so the plan/world
vectors prepended during long-form generation carry no learned information. Measured on held-out
text, even fed unscaled, the prefix had no effect on loss (4.208 with it vs. 4.211 without). `--branching`/`--chapters`
therefore give you the multi-chapter, save-and-resume *machinery* with a plain continuation model
underneath, not planned plots. `--conditioning-scale 0` turns the prefix off entirely.
Training those modules needs paired narrative data (see `asterwhite/training/narrative_objectives.py`).

## Safety
- No safety training, red-teaming, or content filtering of the model itself. The corpus is
  public-domain literature and contains period-typical language and attitudes (including dated
  and offensive depictions of people) that the model can reproduce.
- `asterwhite.safety.PolicyFilter` is a heuristic keyword/regex filter, not a trained classifier.
- Do not expose this model to end users without your own evaluation and safeguards.

## Using and extending it
```bash
python train.py generate --prompt "Chapter One." --max-new-tokens 150     # no other flags needed
python train.py serve                                                       # OpenAI / Ollama / generic API
python train.py evaluate --data your_held_out_text.txt                      # perplexity on your own text

# fine-tune on your own text, starting from these weights (architecture + tokenizer come from the bundle)
python train.py pretrain --load-base pretrained/asterwhite-small --data your_text/ --max-steps 500
python train.py pretrain --load-base pretrained/asterwhite-small --lora --data your_text/ --max-steps 500   # LoRA instead
```
A fine-tune starts at step 0 with the optimizer settings of `--profile` (or the library defaults
if you pass none). For a gentler learning rate, add your own `configs/my_finetune.yaml` with
`extends: pretrained_small` and a lower `training.learning_rate`, and pass `--profile my_finetune`.
Fine-tuning on a corpus this small and this different from 19th-century fiction will mostly
make the model forget its prose style, so keep the data volume and steps modest.

The bundle is a normal checkpoint directory (`latest.json`, `model.json`, `model.safetensors`)
plus `tokenizer/`; the architecture is read from `model.json`, so no profile is required.

## Provenance
- Source checkpoint: `step_2250` of a 3,000-step run of profile `pretrained_small`, exported with
  `train.py export --name step_2250 --ema`.
- `model.safetensors` SHA-256: `6087f9328841e0128918b1a7eaad08382f615f27a9a2d43291e26b755710b9d7`
  (verified on every load).
- Trained in a CPU sandbox with this repository's own tooling; not derived from or initialised
  from any other model's weights.
- Signed release hash: none.
