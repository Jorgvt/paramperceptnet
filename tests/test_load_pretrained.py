from paramperceptnet.pretrained import load_param_pretrained, load_baseline_pretrained

def test_bio_fitted():
    model_name = "ppnet-bio-fitted"
    model, variables = load_param_pretrained(model_name)

def test_fully_trained():
    model_name = "ppnet-fully-trained"
    model, variables = load_param_pretrained(model_name)

def test_baseline():
    model_name = "ppnet-baseline"
    model, variables = load_baseline_pretrained(model_name)
