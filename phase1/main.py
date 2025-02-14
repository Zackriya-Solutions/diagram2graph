from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic_ai import Agent, RunContext
import dotenv
import os
import json
from pydantic_ai.models.anthropic import AnthropicModel
from neo4j import GraphDatabase
from utils import create_edges, create_nodes, check_for_env
from models import (
    DiagramDigitizerDependencies,
    MultiModalLLMService,
    Graph,
)


dotenv.load_dotenv()
app = FastAPI()

anthropic_api_key = check_for_env("ANTHROPIC_API_KEY")
neo4j_uri = check_for_env("NEO4J_URI")
neo4j_password = check_for_env("NEO4J_PASSWORD")

driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", neo4j_password))

diagram_digitizer_agent = Agent(
    AnthropicModel(
        "claude-3-5-sonnet-latest",
        api_key=anthropic_api_key,
    ),
    deps_type=DiagramDigitizerDependencies,
    result_type=Graph,
    retries=3,
    system_prompt="You are a data scientist and you are working on a project to extract information from a diagram in json format. Which is compatiable with knowledge graph databases.Do not change the labels from the image in the output as it should be intact. Consider their shape and translate it's purpose as it is important for the data extraction. only give the json format of the diagram. extract all other information in lowercase and in the same format as the image. But not the labels.",
)


@diagram_digitizer_agent.tool
async def extract_diagram_info(ctx: RunContext[DiagramDigitizerDependencies]) -> Graph:
    """Tool to extract diagram information details from the image"""
    return await ctx.deps.llm_service.perform_task(  # type: ignore
        image_content=ctx.deps.image_content,
        content_type=ctx.deps.content_type,
    )


@app.delete("/delete-all")
def delete_all():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@app.post("/file")
async def upload_file(file: UploadFile = File(...)):
    content_type = file.headers.get("content-type")
    if content_type is None or content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    ]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File type not valid")
    filename = file.filename
    file_content = await file.read()

    try:
        deps = DiagramDigitizerDependencies(
            llm_service=MultiModalLLMService(model="claude-3-5-sonnet-20241022"),
            image_content=file_content,
            content_type=content_type,
        )

        result = await diagram_digitizer_agent.run(
            "Extract the details from the image", deps=deps
        )

        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/{filename}.json", "w") as f:
            f.write(json.dumps(result.data.model_dump(), indent=4))
            # json.dump(result.data.model_dump(), f, indent=4)

    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    finally:
        with driver.session() as session:
            session.execute_write(create_nodes, result.data.model_dump()["nodes"])
            session.execute_write(create_edges, result.data.model_dump()["edges"])

    return result.data.model_dump()
