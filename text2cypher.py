import neo4j
from schema_utils import get_schema
from typing import Literal
import ollama

class Text2Cypher:
    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
        self.dynamic_sections = {}
        self.required_sections = ["question"]
        self.prompt_template = prompt_template

        schema_string = get_schema(driver)
        self.set_prompt_section("schema", schema_string)

    def set_prompt_section(
            self, 
            section: Literal["terminology", "examples", "schema", "question"], 
            value: str):
        
        self.dynamic_sections[section] = value

    def get_full_prompt(self) -> str:
        prompt = self.prompt_template["static"]["instructions"]
        for section in self.prompt_template["dynamic"]:        
            print(f"Adding section: {section}")
            print(f"Dynamic sections value: {self.dynamic_sections}")
            if section in self.dynamic_sections:
                prompt += self.prompt_template["dynamic"][section].format(
                    self.dynamic_sections[section]
                )
        return prompt
    
    def generate_cypher(self):
        for section in self.required_sections:
            if section not in self.dynamic_sections:
                raise ValueError(f"Missing required section: {section}")
        prompt = self.get_full_prompt()
        result = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(f"Generated Cypher: {result}")
        return result['message']['content'].strip()
        

prompt_template = {
    "static": {
        "instructions": """
    Instructions: 
    Generate Cypher statement to query a graph database to get the data to answer the user question below.

    Format instructions:
    Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to 
    construct a Cypher statement.
    Do not include any text except the generated Cypher statement.
    ONLY RESPOND WITH CYPHER, NO CODEBLOCKS.
    Make sure to name RETURN variables as requested in the user question.
    """
    },
    "dynamic": {
        "schema": """
    Graph Database Schema:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided in the schema.
    {}
    """,
        "terminology": """
    Terminology mapping:
    This section is helpful to map terminology between the user question and the graph database schema.
    {}
    """,
        "examples": """
    Examples:
    The following examples provide useful patterns for querying the graph database.
    {}
    """,
        "question": """
    User question: {}
    """,
    },
}    
