# Digitizing process and flow diagrams using Vision Language Models
## Introduction
Building a system for converting images of process diagrams into structured Neo4J-compatible JSON format using AI-based Vision models. The project involves an initial exploration of the same using general purpose models(from Anthropic), followed by fine-tuning a smaller vision language model for improved performance and latency reduction.
## Objectives
  - Extract diagrams information from images
  - Exploring general purpose model(Claude3.5-sonnet)
  - Comparison of image-to-cypher and image-to-json
  - Develop a workflow for image-to-json(convertible to graph) conversion
  - Finetune Vision language model Qwen2.5-vl-3b for image-to-json
  - Implement a prototype for integration
 
## High Level Architecture
![High Level Architecture](docs/high_level.png)

## Conclusion
This document provides a structured workflow for digitizing process diagrams by converting them to knowledge graph representation, for future usage, especially for AI based development
