import json


class GraphLoader:

    def __init__(self, schema_config: dict) -> None:
        self.schema_config = schema_config

    @staticmethod
    def from_config_file(path: str) -> "GraphLoader":
        with open(path, "r", encoding="utf-8") as file:
            schema_config = json.load(file)
        return GraphLoader(schema_config=schema_config)


if __name__ == "__main__":

    import os
    from src.configs.config import ConfigPath

    file_path = os.path.join(ConfigPath.KG_CONFIG_DIR, "schema_config.json")
    graph_loader = GraphLoader.from_config_file(path=file_path)
    print("graph loader initiated successfully!")
