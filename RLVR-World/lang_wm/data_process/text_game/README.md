Download the [GPT-simulator](https://github.com/cognitiveailab/GPT-simulator) repository. According to its instructions, unzip ``data/train.zip`` and ``data/test.zip`` to obtain ``train.jsonl`` and ``test.jsonl``, and place them in the original directory (i.e., under the ``data/`` folder).

Place the two processing scripts ``data_process_for_text_game_simulator/process_jsonl.py`` and ``data_process_for_text_game_simulator/process_jsonl_train.py`` into the ``experiments/`` folder of the GPT-simulator. You should see a file named ``quest_gpt.py`` in the same directory.

Modify these two scripts as follows:

1. Line 12: Update the path to point to your local installation of the GPT-simulator so that relative paths can be correctly recognized.

2. In ``process_jsonl.py`` (line 466) and ``process_jsonl_train.py`` (line 593), update the data-saving paths to save files to your desired location.

Run the following commands to generate the raw data:
```commandline
python experiments/process_jsonl.py
python experiments/process_jsonl_train.py gold
python experiments/process_jsonl_train.py no_gold
python experiments/process_jsonl_train.py all
```

The raw data does not contain the reasoning process, so you need to use Deepseek-R1 to generate pre-training data. First, move the files ``calculate_r1_response.py``, ``call_deepseek_r1.py``, ``generate_sft_data.py``, and ``text_game.py`` from the ``data_process_for_text_game_simulator/`` folder to the root directory of the GPT-simulator repository (you should see the ``experiments/`` folder in the same directory).

Many file paths need to be specified in these three files (e.g., the path to the Qwen model, the path to save the generated data, etc.). These places are marked with ``TODO``. Please locate all ``TODO`` comments and update the paths accordingly.

Finally, you need to obtain a Deepseek API key, and then run the following commands:
```commandline
python call_deepseek_r1.py --start_index 0 --end_index 20000 --api_key your_api_key
python calculate_r1_response.py
python generate_sft_data.py
```
