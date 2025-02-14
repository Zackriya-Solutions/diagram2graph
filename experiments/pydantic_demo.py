from pydantic import BaseModel, Field
from typing import List, Literal
import dotenv
import base64
from anthropic import Anthropic
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from pydantic_ai.models.anthropic import AnthropicModel
# from openai import OpenAI


anthropic_api_key = dotenv.get_key(dotenv_path=".env", key_to_get="ANTHROPIC_API_KEY")
if anthropic_api_key is None:
    Exception("APi key not available")


class Node(BaseModel):
    """Structure of a node in the diagram"""

    id: str = Field(description="Id of the Node")
    type_of_node: Literal["process", "decision", "delay", "terminator"] | str = Field(
        description="The type of node"
    )
    label: str = Field(description="Label of the node")


class Edge(BaseModel):
    """Structure of a edge in the diagram"""

    from_: str = Field(description="The ID of the edge's starting node")
    to: str = Field(description="The ID of edge's end node")
    type_of_edge: Literal["dashed", "solid"] | str = Field(
        description="The type of edge, visually"
    )


class Graph(BaseModel):
    """Structure of the graph representing the diagram"""

    nodes: List[Node] = Field(description="Nodes from the diagram")
    edges: List[Edge] = Field(description="Edges from the diagram")


class MultiModalLLMService:
    """Service to interact with Anthropic multimodal LLMs."""

    def __init__(self, model: str):
        self.client = Anthropic(api_key=anthropic_api_key)
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def perform_task(
        self, image_path: str, response_model: type, max_tokens: int = 1000
    ):
        """Send an image and prompt to the LLM and return structured output."""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                    },
                ],
            }
        ]
        response = self.client.messages.create(  # type: ignore
            model=self.model,
            max_tokens=max_tokens,
            messages=message_list,
            # response_format=response_model, only supports OpenAI
        )
        return response.content[0].text


@dataclass
class DiagramDigitizerDependencies:
    llm_service: MultiModalLLMService
    diagram_path: str


diagram_digitizer_agent = Agent(
    AnthropicModel("claude-3-5-sonnet-latest", api_key=anthropic_api_key),
    deps_type=DiagramDigitizerDependencies,
    result_type=Graph,
    system_prompt="You are a data scientist and you are working on a project to extract information from a diagram in json format. Which is compatiable with knowledge graph databases. Consider their shape and translate it's purpose(process, decision, etc.) as it is important for the data extraction. only give the json format of the diagram.",
)


@diagram_digitizer_agent.tool
async def extract_diagram_info(ctx: RunContext[DiagramDigitizerDependencies]) -> Graph:
    """Tool to extract diagram information details from the image"""
    return await ctx.deps.llm_service.perform_task(
        image_path=ctx.deps.diagram_path, response_model=Graph
    )


async def main():
    deps = DiagramDigitizerDependencies(
        llm_service=MultiModalLLMService(model="claude-3-5-sonnet-20241022"),
        diagram_path="./images/image.png",
    )

    result = await diagram_digitizer_agent.run(
        "Extract the details from the image", deps=deps
    )
    print("Structured Result:", result.data)
    print("=" * 100)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
