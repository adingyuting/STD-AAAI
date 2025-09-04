# The official implementation of STD-PLM.

### Train and Test the STD-PLM

```bash
python code/STD-PLM/src/main.py --dataset D1
```

### Custom datasets

For the built‑in raw table datasets (`D1` beijingPM2.5, `D2` beijingCO, `D3` sea, `D4` cailiCO), place your `data.*` and `position.*` files under `datasets/<name>/` and run:

```bash
python code/STD-PLM/src/main.py --dataset D1
```

When `data_path` or `adj_filename` are omitted, the loader automatically locates `data.*` and `position.*` under the dataset folder and uses the position file to construct the adjacency matrix.

### Arguments

```
--lora : Fine-tune the PLM with lora
--ln_grad : Train the layernorm of PLM
--wo_conloss  : Removing the constraint loss function
--sandglassAttn : Introducing the SGA module.
--time_token : Add time token
--model plm : Specify the PLM to use
--llm_layers layers : Specify the layers of PLM
--few_shot ratio    :   Specify the ratio of few-shot
--zero_shot :   zero-shot
--from_pretrained_model model.pth   : Specify the trained model weights
--task task : Specify the task
```
