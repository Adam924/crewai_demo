# Agentic RAG Chatbot

This project implements an Agentic RAG (Retrieval-Augmented Generation) chatbot using CrewAI, Chonkie, and DeepEval. This project is an extension of the template given by CrewAI, and serves to integrate RAG with CrewAI.

## Initial Thoughts
This was my first time implementing an Agentic RAG chatbot, and also my first time using any of the frameworks. Initial, this project was very daunting to me and I didn't know where to start. However, as I looked through the given documentation I felt I had a general direction. I broke this project into four main steps:

- **Chunking with Chonkie**: Processing and chunking data is the core of RAG so getting this operational was first.
- **Qdrant Vector Database**: Setting up the vector database with Qdrant and testing the similarity score so that it adequately matches the queries.
- **CrewAI Framework**: Get a basic agent framework running and have it use the data fed from Qdrant
- **DeepEval Evaluation**: To test RAG metrics

I was able to set up Chonkie and Qdrant up relatively easily, but I am currently roadblocked with integrating it with CrewAI. Unfortunately this is still not functionally and I hope to get this fixed soon.

## Setup

1. Start the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Qdrant:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

3. Set up environment variables:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

1. Process documents:
   ```bash
   python -m src.crewai_demo.main --process knowledge/docs
   ```

2. Test retrieval:
   ```bash
   python -m src.crewai_demo.main --query "What is the objective of this project?" --test
   ```

3. Run the chatbot:
   ```bash
   python -m src.crewai_demo.main --query "What is the objective of this project?"
   ```

## Architecture

### Document Chunking with Chonkie

The system uses Chonkie's NeuralChunker for semantic document chunking:

```python
from chonkie import NeuralChunker

class DocumentChunker:
    def __init__(self, model="mirth/chonky_modernbert_base_1", min_characters=100):
        self.chunker = NeuralChunker(
            model=model,
            device_map="cpu",
            min_characters_per_chunk=min_characters,
            return_type="chunks"
        )
```

NeuralChunker was selected because it:
- Creates semantically meaningful chunks that preserve context
- Uses neural models to understand document structure
- Produces better retrieval results compared to fixed-size chunking

### Vector Storage with Qdrant

The system uses Qdrant for vector storage and retrieval:

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, collection_name="document_chunks", model_name="all-MiniLM-L6-v2"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
```

Qdrant was chosen for its:
- Efficient vector similarity search
- Simple deployment options
- Good performance with medium-sized vector collections

### Agent Framework with CrewAI

The system uses CrewAI to manage specialized agents:

```python
from crewai import Agent, Crew, Process, Task

# Example agent definitions
researcher = Agent(
    name="Researcher",
    role="Information Retrieval Specialist",
    goal="Find relevant information",
    tools=[retrieval_tool]
)

reporting_analyst = Agent(
    name="Reporting Analyst",
    role="Content Generator",
    goal="Create comprehensive responses"
)
```

CrewAI enables:
- Task sequencing with dependencies
- Agent specialization for different parts of the RAG pipeline
- Tool integration for external functionality

### Evaluation with DeepEval

The system uses DeepEval to evaluate RAG performance:

```python
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
```

These metrics assess:
- Retrieval quality (contextual precision, recall, relevancy)
- Generation quality (answer relevancy, faithfulness)

## Implementation Challenges

1. **Chunking Strategy**: Finding an optimal balance between chunk size and semantic coherence.
2. **Tool Integration**: Properly formatting tools for CrewAI agent integration.
3. **Configuration Management**: Handling CrewAI's YAML-based configuration system for agents and tasks.
4. **Pipeline Coordination**: Ensuring smooth data flow between different components.

## Project Structure

```
crewai_demo/
├── knowledge/            # Document storage
│   ├── docs/             # Source documents
│   └── chunks.json       # Processed chunks
├── src/
│   ├── crewai_demo/
│   │   ├── config/       # CrewAI configurations (agents.yaml & tasks.yaml)
│   │   ├── tools/        # RAG implementation (Chonkie and Qdrant)
│   │   ├── deepeval/     # Evaluation metrics
│   │   ├── crew.py       # CrewAI setup
│   │   └── main.py       # Main application
├── tests/                # Test modules (I can provide my test files if desired)
└── requirements.txt      # Dependencies
```

## Future Improvements

1. **Incremental Learning**: Add feedback loops to improve retrieval based on user interactions.
2. **Advanced Agent Behaviors**: Implement more sophisticated agent decision-making.
3. **Enhanced Evaluation**: Add user-focused metrics for response quality assessment.
4. **Web Interface**: Create a web UI for easier interaction with the chatbot.

## Final Thoughts

This project was a wonderful introduction to RAG and agentic AI. Personally, this was my first time using any of these frameworks so being able to get hands on experience and actually getting an understanding of how RAG functions and how it can be used with AI was very insightful. This project has further ignited my passion for AI and I hope that I can work on projects similar to this again in the future.