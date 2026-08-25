# Catopter

Robust and efficient provably secure steganography in asymmetric communication scenarios.

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Put language model weights under the `llm/` directory, for example:

   ```bash
   llm/Llama-3.1-8B/
   ```

   Alternatively, pass any Hugging Face model name directly to `--model_path`.

3. Run one method:

   ```bash
   python main.py --method catopter --model_path ./llm/Llama-3.1-8B --message_length 64
   ```

   Supported methods: `catopter`, `sparsamp`, `ars`, `anstega`.

## Reproducing Experiments

Open `exp/results.ipynb` with Jupyter.

The notebook:

- loads existing summaries from `exp/results/exp*/exp*_summary.json`;
- prints the main tables and plots;
- automatically runs any missing experiment and writes outputs to `exp/results/`.

Experiment data and figures are stored in:

- `exp/results/`
- `exp/figures/`
- `exp/data/`
