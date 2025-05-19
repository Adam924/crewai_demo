#!/usr/bin/env python
import sys
import warnings
import argparse
import os
import json
from datetime import datetime

from src.crewai_demo.crew import CrewaiDemo
from src.crewai_demo.tools.chunker import DocumentChunker
from src.crewai_demo.tools.rag_tools import RAGTools

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def process_documents(doc_dir="knowledge/docs", force=False):
    """Process documents and store in vector database"""
    # Check if chunks already exist
    chunks_path = "knowledge/chunks.json"
    chunks = []
    
    if os.path.exists(chunks_path) and not force:
        print(f"Chunks file exists at {chunks_path}. Use --force to reprocess.")
        try:
            with open(chunks_path, 'r') as f:
                chunks = json.load(f)
            print(f"Loaded {len(chunks)} chunks from file.")
        except json.JSONDecodeError:
            print(f"Error: Chunks file is corrupted. Will reprocess documents.")
            force = True
        except Exception as e:
            print(f"Error loading chunks file: {e}. Will reprocess documents.")
            force = True
    
    if force or not os.path.exists(chunks_path) or not chunks:
        print("Processing documents...")
        # Chunk documents
        chunker = DocumentChunker()
        chunks = chunker.chunk_directory(doc_dir, chunks_path)
    
    # Store in vector database
    from src.crewai_demo.tools.vectordb import VectorDB
    vector_db = VectorDB()
    vector_db.store_chunks(chunks)
    
    return len(chunks)

def run():
    """
    Run the crew.
    """
    parser = argparse.ArgumentParser(description="Agentic RAG Chatbot")
    parser.add_argument("--process", help="Process documents in the specified directory", metavar="DIR")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of documents")
    parser.add_argument("--query", type=str, help="The query to answer")
    parser.add_argument("--test", action="store_true", help="Test retrieval only")
    args = parser.parse_args()
    
    # Process documents if requested
    if args.process:
        num_chunks = process_documents(args.process, args.force)
        print(f"Processed documents into {num_chunks} chunks")
        return
    
    # Test retrieval if requested
    if args.test and args.query:
        rag_tools = RAGTools(collection_name="document_chunks")
        results = rag_tools.get_relevant_chunks(args.query, top_k=2)
        
        print(f"\nQuery: {args.query}")
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"Result {i+1} (Score: {result['score']:.4f}):")
                print(f"Text: {result['text'][:100]}...")
        else:
            print("No results found")
        return
    
    # If query provided, include it in inputs
    inputs = {
        'topic': 'AI RAG Systems',
        'current_year': str(datetime.now().year)
    }
    
    if args.query:
        # Get retrieval context for the query
        rag_tools = RAGTools(collection_name="document_chunks")
        retrieval_context = rag_tools.retrieve_context(args.query)
        
        # Add query and context to inputs
        inputs['query'] = args.query
        inputs['retrieval_context'] = retrieval_context
        
        print(f"Retrieved context for query: '{args.query}'")
    
    try:
        # Create and run the crew
        crew_instance = CrewaiDemo()
        result = crew_instance.crew().kickoff(inputs=inputs)
        print("\n--- Response ---")
        print(result)
    except Exception as e:
        print(f"Error running crew: {e}")
        import traceback
        traceback.print_exc()

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        CrewaiDemo().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CrewaiDemo().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        CrewaiDemo().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()