# Graphormer-Guided Task Planning (GGTP)

## Reproduce Experiments

Follow these steps to generate and train the dataset.

### 1️⃣ Extract AI2-THOR Object Types  
```bash
python experiments/data/get_all_ai2thor_objects.py
```
- Generates `data/ai2thor_object_types.json`

### 2️⃣ Define Human Label Data  
Modify or create:  
```bash
experiments/data/human_entities.json
```

### 3️⃣ Generate Danger Information  
```bash
python experiments/data/danger_info_gen.py
```
- Outputs: `experiments/data/danger_info.json`

### 4️⃣ Process Raw Data  
```bash
python experiments/data/process_raw_data.py
```
- Finalizes danger labels.

### 5️⃣ Build Dataset  
```bash
python experiments/data/train_data_gen.py
```
- Outputs: `experiments/data/graph_dataset.json`

### 6️⃣ Train Graphormer Model  
```bash
python ./model/train.py
```
- Generates the trained model.

### 7️⃣ Run Experiments  
```bash
python experiments/src/run_all_scenes.py
```
- Outputs: `./experiments/data/{method}_task_data_{difficulty}.json`

### 8️⃣ Evaluation  
```bash
python experiments/src/evaluation.py
```
