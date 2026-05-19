# Netflix-llm-based-model
# LLM-based-feature-generation

This is my coursework project for the 4ES105 course. The idea is to use a large language model to automatically come up with useful features from text, and then use those features to train a regular ML classifier.

I chose the Netflix titles dataset because it's something I actually find interesting — trying to figure out whether a title is a Movie or a TV Show just from its description is harder than it sounds. A lot of descriptions sound the same regardless of format.



## What I did

The main task was to take raw text descriptions and let the LLM figure out what features might be useful — things like whether the story seems self-contained, whether it mentions multiple seasons, whether the plot has a single clear resolution. The LLM then goes through all the descriptions and scores each one on those features. That gives you a nice tabular dataset you can plug into any standard classifier.

I compared this against a plain TF-IDF + Logistic Regression baseline to see how interpretable features hold up against a model that has access to every word.

For the extra contribution tasks I extended the library in a few ways:
- Wrapped the whole pipeline as a proper sklearn Transformer so it fits into a Pipeline like anything else
- Added batch processing so it doesn't call the LLM once per file (that was painfully slow on 8000+ titles)
- Added disk caching so you don't have to redo everything if the notebook crashes halfway through
- Made it work with more than 2 classes — tested with Kids / Teen / Adult based on Netflix rating
- Found and reported a bug where the CSV output breaks if the LLM returns a value containing a comma



## Dataset

Netflix Movies and TV Shows — about 8,800 titles. I used the `description` column as input and `type` (Movie or TV Show) as the label.

Source: https://www.kaggle.com/datasets/shivamb/netflix-shows



## Files

- `HW LLM-based feature generation.ipynb` — main notebook, covers the basic task
- `sklearn Transformer + Batch Processing + Multi-Class.ipynb` — contribution tasks
- `netflix_titles.csv` — dataset
- `discovered_text_features.json` — features the LLM came up with during discovery
- `all_feature_values.csv` — the generated feature matrix used for training



## How to run

1. Install dependencies:
   ```
   pip install llm-feature-gen scikit-learn pandas matplotlib
   ```
2. Put `netflix_titles.csv` in the same folder as the notebooks
3. Run the main notebook top to bottom — it connects to the VSE LiteLLM server so you need university network access or VPN
4. The sklearn notebook is separate and covers the contribution tasks



## Results

The TF-IDF baseline gets around 82% accuracy — not surprising since it sees every word. The LLM feature tree gets around 78%, which is a bit lower, but you can actually open the tree and read exactly why it classified something as a Movie vs a TV Show. That interpretability is the whole point.



## A few things I ran into

- The `qwen3.6-35b` model outputs `<think>...</think>` blocks before the actual JSON response, which breaks the library's parser. I patched it directly in the notebook.
- I capped the sample at 50 train + 30 test per class to keep the runtime reasonable. Running the full 8,800 titles would take several hours.
- The library currently assumes exactly 2 classes in a few places — one of my contribution PRs fixes that.
