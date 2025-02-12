from typing import List

from pydantic import BaseModel, Field, create_model


class ModelFactory:

    def create_dynamic_base_model(
        self, model_name: str, properties: dict, model_description: str = None
    ) -> BaseModel:
        dynamic_base_model = create_model(
            model_name, __doc__=model_description, **properties
        )
        return dynamic_base_model

    def _construct_property_field(self, name, type, description) -> dict:
        return {name: (type, Field(description=description))}

    def construct_property_fields(self, properties: List[dict]):
        properties_dict = {}
        for property in properties:
            properties_dict.update(
                self._construct_property_field(
                    name=property["name"],
                    type=property["type"],
                    description=property.get("description"),
                )
            )

        return properties_dict
