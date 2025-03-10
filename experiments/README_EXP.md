# Graphormer-Guided Task Planning (GGTP)

## Reproduce Experiments

To reproduce the dataset labeling process, follow these steps:

### 1. Extract AI2-THOR Object Types  
Run the following script to generate the list of AI2-THOR object types:  
```bash
python experiments/data/get_all_ai2thor_objects.py
```
This will create `data/ai2thor_object_types.json`, which contains all objects in AI2-THOR.

### 2. Define Custom Task Data  
Modify or create ` experiments/data/graphormer_task_data_simple.json` to specify task-related object interactions.

### 3. Generate Danger Information  
Run the following script to annotate objects and people with safety labels:  
```bash
python  experiments/data/danger_info_gen.py
```
This will generate raw danger information.

### 4. Process Raw Data  
To finalize the labeled dataset, run:
```bash
python  experiments/data/process_raw_data.py
```
The processed danger labels will be saved in ` experiments/data/danger_info.json`.

### 5. Build Dataset
```bash
python  experiments/data/train_data_gen.py
```
This will generate ` experiments/data/graph_dataset.json`

### 6. Train Graphormer Model
```bash
python  ./model/train.py
```
This will generate the model file.
### 7. Run Experiments
```bash
python  experiments/src/run_all_scenes.py
```
This will generate "./experiments/data/{method}_task_data_{difficuty}.json"

### 8. Evaluation
```bash
python  experiments/src/evaluation.py
```
