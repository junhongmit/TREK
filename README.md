# Temporal Reasoning over Evolving Knowledge Graphs
This is the accompany code for the paper "Temporal Reasoning over Evolving Knowledge Graphs".

![overview](img/overview.png)

## Environment Setup
The easiest way to setup the environment is to install the `vllm` library, which will automatically install all the required dependency. We also have additional dependencies defined in the `requirements.txt` located under the root directory. Follow the listed commands to install a local conda environment named `vllm`:
```
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
pip install -r requirements.txt
```
After this, you will need to fill in the local environment variables. They are automatically loaded from a `.env` file. Copy the `.env_template` and rename it to be `.env` file, and fill in the settings. 
> 1. The `API_BASE` usually ends with "/v1". For example, `API_BASE="http://localhost:7878/v1"`. 
> 2. The `API_KEY` is usually `"DUMMY"` unless you are really using the OpenAI API.

> :bulb: **You can choose different environment config file at runtime by prepending an environment variable: `ENV_FILE=path/to/your/.env/file`. This is very useful to manage different LLM models and KG instances.**

## Set up the Neo4j graph database
The framework heavily rely on Neo4j graph database to perform operation. Download the neo4j database (version 5.26.3) to current folder:
```
wget "https://neo4j.com/artifact.php?name=neo4j-community-5.26.3-unix.tar.gz" -O neo4j.tar.gz
tar -xvzf neo4j.tar.gz
mv neo4j-community-*/ neo4j/
cd neo4j
```
Set the Neo4j password by following:
```
bin/neo4j-admin dbms set-initial-password password
```
> If you encounter no Java Runtime found error, please install it following this instruction or directly install it through the [Oracle website](https://www.oracle.com/java/technologies/downloads/#jdk21-mac):
> ```
> wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz
> tar -xvzf jdk-21_linux-x64_bin.tar.gz
> mv jdk-21.0.6/ java21/
> ```
> Then set the Java Path in your bash file:
> ```
> echo 'export JAVA_HOME=$HOME/java21' >> ~/.bashrc
> echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
> source ~/.bashrc
> ```

### Install the APOC extension
Navigate to your `neo4j` folder, there is a `neo4j/labs` folder inside. Copy the `apoc-5.26.3-core.jar` inside to `neo4j/plugins` folder.
```
cp neo4j/labs/apoc-5.26.3-core.jar neo4j/plugins/apoc-5.26.3-core.jar
```

### Start/Stop the KG
During running the KG Update stage or LLM Inference stage, a normal running KG is required. It's recommended to run this within a [terminal multiplexer](https://github.com/tmux/tmux/wiki).  Navigate to `neo4j` folder and start/stop the local server: 
```
cd neo4j
bin/neo4j start
bin/neo4j stop
bin/neo4j restart
```

### Optional: View/Interact with the KG from Local
Depends on your settings, follow one of these steps:
- If you have set up everything on your local machine, then you can directly view and interact with the KG in a browser.
- If you set up everything on a server, you can create a tunnel from the GPU cluster to your local machine:
    ```
    ssh -L 7474:localhost:7474 -L 7687:localhost:7687 your_username@remote_cluster_address
    ```

Open your browser and head to Neo4j console located at `http://localhost:7474/browser/`.
Following the prompt and enter the default user name as `neo4j` and password as `password`.

## Dataset Preparation
We evaluate the framework using [CRAG benchmark](https://arxiv.org/pdf/2406.04744). First, git clone the [CRAG Benchmark repo](https://github.com/facebookresearch/CRAG), and copy the `CRAG/mock_api/movie` and `CRAG/mock_api/music` folders into the current `dataset` folder (they contain the KG that is too big to upload to Github). Note: you may need to install [Git LFS](https://git-lfs.com/) to properly git clone all the raw datasets.

Secondly, download the Question Answering (QA) dataset from [here](https://drive.google.com/drive/folders/1-iI4d6HX_65W6EmtoB0JAo4-dXXolxXt?usp=sharing) and put the two files under your specifified `DATASET_PATH` path in the `.env` fild.

(They are the QA pairs filtered by the particular domains from the [CRAG Benchmark](https://github.com/facebookresearch/CRAG). The dataset I am currently working on is the [QA Pairs & Retrieval Contents](https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2?download=))

> :warning: **Remember to adjust the `DATASET_PATH` located in `.env` to your dataset location.**

## Import the Dataset into Neo4j KG
1. Run `python -m run.run_kg_preprocess` and it will automatic import the `movie` KG to Neo4j.
2. Then run `python -m run.run_kg_embed` to generate the entity/relation embeddings.

## Run KG Update stage
Run `python -m run.run_kg_update` to start the KG update process. You can monitor the real-time progress by open the web page located at `visualization/kg_progress.html`.

> :warning: **You need to run `python -m visualization.progress_server --dataset <dataset_name> --model <model_name>` under the root directory to start the data pushing backend first.**

## Run LLM Inference stage
Run the following, where `model` name can be one of the "io", "rag", "kg", "our".
Resutls will be automatically stored under the `results` folder. Evaluation is also automatically run after the inference. You can monitor the real-time progress by open the web page located at `visualization/qa_progress.html`.
```
`python -m run.run_qa --dataset [movie, sports] --model [io, rag, kg, our]`
```

> :warning: **You need to run `python -m visualization.progress_server --dataset <dataset_name> --model <model_name>` under the root directory to start the data pushing backend first.**

# Run evaluation
Evaluation is always automatically run at the end of each experiment. But if you can't help to wait till the last moment,
you can manually perform evaluation on the intermediate resutls in an ongoing experiment using:
```
python -m run.run_eval --dataset [movie, sports] --model [io, rag, kg, our]
```

You can also re-evaluate a previous evaluated results:
```
ENV_FILE=.env.eval python -m run.run_eval --reeval results/sports_llama-3-3-70b/io_sports_results-0.json
```


# Known Issues

- **Q**: I am using a Mac OS, and the BlingFire package complains:

  ```
  OSError: dlopen(/Users/XXXX/conda/envs/vllm/lib/python3.12/site-packages/blingfire/libblingfiretokdll.dylib, 0x0006): tried: '/Users/XXXX/conda/envs/vllm/lib/python3.12/site-packages/blingfire/libblingfiretokdll.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/XXXX/conda/envs/vllm/lib/python3.12/site-packages/blingfire/libblingfiretokdll.dylib' (no such file), '/Users/XXXX/conda/envs/vllm/lib/python3.12/site-packages/blingfire/libblingfiretokdll.dylib' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64'))
  ```

- **A**: The BlingFire package doesn't support Apple Silicon natively. You will need to manually [build it from source](https://github.com/microsoft/BlingFire/blob/master/nuget/readme.md).

  Steps to fix:
  
  1. Take note of the location mentioned in the error message where the `.dylib` file is located.
  
  2. Clone the BlingFire repository and build it manually (you may need to install CMake — [installation guide here](https://gist.github.com/fscm/29fd23093221cf4d96ccfaac5a1a5c90)):

     ```bash
     git clone https://github.com/microsoft/BlingFire.git
     cd BlingFire
     mkdir Release
     cd Release
     cmake ..
     make
     ```

  3. After the build completes successfully, copy the newly built `.dylib` file from the `Release` folder to the location you noted earlier.