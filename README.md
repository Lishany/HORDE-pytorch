## Re-implementation of HORDE (harmonized representation learning on dynamic EHR)
Here is the PyTorch version for re-implementation of HORDE.

The HORDE article:

```
@article {lee2020harmonized,
title = {Harmonized representation learning on dynamic EHR graphs},
author = {Lee, Dongha and Jiang, Xiaoqian and Yu, Hwanjo},
volume = {106},
year = {2020},
journal = {Journal of biomedical informatics},
pages = {103426}
}
```

### Data Description
data directory: ./data/

Data structure: list of quads (patient id, visit id, list of events in the visit, list of concepts in the visit)

### Run model
*run process_data.py*:  Pre-process data and obtain ctxpairs.npy (edges in graph), patients.npy (training data) and testpatients.npy (test data)

*run main.py*: train and test model
