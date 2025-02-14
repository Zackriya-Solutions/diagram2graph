import base64
from dataclasses import dataclass
from typing import List, Literal
from anthropic import Anthropic
import dotenv
import os
from pydantic import BaseModel, Field

dotenv.load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_api_key is None:
    Exception("APi key not available")


class Node(BaseModel):
    """Structure of a node in the diagram"""

    id: str = Field(description="Id of the Node")
    type_of_node: Literal["process", "decision", "delay", "terminator", "start"] = (
        Field(description="The type of node")
    )
    shape: Literal["task", "gateway", "start_event", "end_event", "data_store"] = Field(
        description="The shape of the node visulally"
    )
    label: str = Field(description="Label of the node")


class Edge(BaseModel):
    """Structure of a edge in the diagram"""

    from_: str = Field(description="The ID of the edge's starting node")
    from_type: Literal["process", "decision", "delay", "terminator", "start"] = Field(
        description="The type of the edge's starting node"
    )
    from_label: str = Field(description="The label of the edge's starting node")
    to: str = Field(description="The ID of edge's end node")
    to_type: Literal["process", "decision", "delay", "terminator", "start"] = Field(
        description="The type of the edge's end node"
    )
    to_label: str = Field(description="The label of the edge's end node")
    type_of_edge: Literal["dashed", "solid"] = Field(
        default="solid", description="The type of edge, visually"
    )
    relationship_value: str = Field(
        default="",
        description="The label of the relationship in the image, if present(e.g., 'yes', 'no')",
    )
    relationship_type: Literal["follows", "branches", "depends_on"] = Field(
        default="follows",
        description="Semantic type of the relationship (e.g., 'follows', 'branches', 'depends_on')",
    )


class Graph(BaseModel):
    """Structure of the graph representing the diagram"""

    nodes: List[Node] = Field(description="Nodes from the diagram")
    edges: List[Edge] = Field(description="Edges from the diagram")


class MultiModalLLMService:
    """Service to interact with Anthropic multimodal LLMs."""

    def __init__(self, model: str):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model = model

    async def perform_task(
        self,
        image_content: bytes,
        content_type: str | None,
        max_tokens: int = 1000,
    ):
        """Send an image and prompt to the LLM and return structured output."""
        base64_image = base64.b64encode(image_content).decode("utf-8")

        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content_type,
                            "data": base64_image,
                        },
                    },
                ],
            }
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=message_list,
        )
        return response.content[0].text


@dataclass
class DiagramDigitizerDependencies:
    llm_service: MultiModalLLMService
    image_content: bytes
    content_type: str
