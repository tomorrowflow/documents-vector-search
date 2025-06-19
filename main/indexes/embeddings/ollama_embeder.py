import os
import requests

class OllamaEmbedder:
    def __init__(self, model_name=None):
        """
        Initialize the OllamaEmbedder with environment variables for configuration.

        The following environment variables can be set:
        - OLLAMA_HOST: The host for the Ollama API (default: http://localhost:11434)
        - OLLAMA_MODEL: The embedding model to use (default: nomic-embed-text)

        Args:
            model_name (str, optional): The model name to override the environment variable.
        """
        # Get the model name from environment variable or use the provided one or default
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "nomic-embed-text")

        # Get the API host from environment variable or use default
        self.api_url = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/embed"

    def embed(self, text):
        """
        Generate embeddings for the given text using Ollama's API.

        Args:
            text (str): The input text to generate embeddings for.

        Returns:
            list of float: The embedding vector.
        """
        response = requests.post(
            self.api_url,
            json={
                "model": self.model_name,
                "prompt": text
            }
        )

        if response.status_code != 200:
            raise Exception(f"Error from Ollama API: {response.status_code} - {response.text}")

        result = response.json()
        return result.get("embedding", [])

    def get_number_of_dimensions(self):
        """
        Get the number of dimensions for the embeddings.

        For Ollama API, we'll assume a fixed dimensionality based on the model.
        In a real implementation, this might be dynamically determined.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        # This is a placeholder. In a real implementation, you might
        # query the API for the model's dimensionality or have it configured.
        # For example purposes, we are using 768 dimensions for nomic-embed-text model.
        return 768