from toolbox import compute_rdm
from torch.utils.data import DataLoader
from IPython.display import display
import pandas as pd, torch
from core import MODEL_CLASSES, SantoroDataset, extract_activations
from toolbox import *


dataset = SantoroDataset()
brain_responses = dataset.brain_responses.detach().cpu().numpy()
brain_rdm = compute_rdm(brain_responses)


dataloader = DataLoader(dataset, batch_size=16, shuffle=False)


models = {}
activations = {}
rdms = {}
upper_matrix = {}

for name, model_class in MODEL_CLASSES.items():
    models[name] = {}
    for version in ["untrained", "trained"]:
        model = model_class(num_classes=50)
        if version == "trained":
            weights = torch.load(
                f"models/{name}_run0_best.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(weights)
        model.eval()
        models[name][version] = model


for name, versions in models.items():
    activations[name] = {}
    for version, model in versions.items():
        activations[name][version] = extract_activations(dataloader, model) 
        for layer_name, responses in  activations[name][version].items():
            patterns  = responses.detach().cpu().numpy()
            patterns = patterns.reshape(patterns.shape[0], -1)
            current_rdm = compute_rdm(patterns)
            rdms[name, version, layer_name] = current_rdm
            upper_matrix[name, version, layer_name] = upper_triangle(current_rdm)



# calculate the coorelation 

results = []
for (name, version, layer_name), model_rdm in rdms.items():
    correlation = compare_rdms(brain_rdm, model_rdm)

    results.append({
        "model": name,
        "version": version,
        "layer": layer_name,
        "correlation": correlation,
    })

results_df = pd.DataFrame(results)
display(results_df)




