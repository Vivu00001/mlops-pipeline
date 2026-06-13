import yaml

def load_params():

    with open(
        "params.yaml",
        "r",
        encoding="utf-8-sig"
    ) as f:

        return yaml.safe_load(f)