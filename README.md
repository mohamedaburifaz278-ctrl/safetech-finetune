# Fine-tune llama3.2:1b as the Safe Tech assistant

Everything is ready. The training data (64 examples about Safe Tech Solutions,
Rifa persona, services, products, pricing replies) is already embedded inside
the notebook — you upload ONE file and press Run all.

## Steps

1. Open https://colab.research.google.com (any Google account)
2. **File → Upload notebook** → choose `SafeTech_Finetune_Colab.ipynb` from this folder
3. **Runtime → Change runtime type → T4 GPU** → Save
4. **Runtime → Run all** and wait ~20–30 minutes
   - Step 6 in the notebook shows test answers so you can see it learned
   - At the end, `safetech-ft.gguf` (~0.8 GB) downloads automatically
5. Move the downloaded file into this folder:
   `~/Documents/Safe Tech Solutions/finetune/`
6. In a terminal:
   ```bash
   cd ~/Documents/"Safe Tech Solutions"/finetune
   ollama create safetech-ft -f Modelfile
   ollama run safetech-ft
   ```
7. Ask it "Who are you?" — it should answer as Rifa with real Safe Tech facts.

## Files in this folder

| File | What it is |
|---|---|
| `SafeTech_Finetune_Colab.ipynb` | The one file you upload to Colab |
| `dataset.jsonl` | The 64 training examples (also embedded in the notebook) |
| `make_dataset.py` | Script that generated the dataset — edit + rerun to add examples |
| `build_notebook.py` | Rebuilds the notebook after you change the dataset |
| `Modelfile` | Used in step 6 to import the trained model into Ollama |

## To improve it later

Add more (question, answer) pairs in `make_dataset.py`, then:
```bash
python3 make_dataset.py && python3 build_notebook.py
```
and repeat the Colab steps. More varied examples = better model.
