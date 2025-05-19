from chonkie import NeuralChunker
import os
import json

class DocumentChunker:
    """Document chunker using Chonkie's NeuralChunker for semantic chunking"""
    
    def __init__(self, model="mirth/chonky_modernbert_base_1", min_characters=100):
        """Initialize the chunker with NeuralChunker"""
        self.chunker = NeuralChunker(
            model=model,
            device_map="cpu",
            min_characters_per_chunk=min_characters,
            return_type="chunks"
        )
    
    def chunk_document(self, document_path):
        """Chunk a single document into semantic chunks"""
        # Read the document
        with open(document_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create chunks using Chonkie's NeuralChunker
        raw_chunks = self.chunker.chunk(text)
        
        # Convert to standard format
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            if not isinstance(chunk_text, str):
                chunk_text = str(chunk_text)
                
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": document_path,
                    "chunk_index": i
                }
            })
        
        return chunks
    
    def chunk_directory(self, dir_path="knowledge/docs", output_path="knowledge/chunks.json"):
        """Chunk all documents in a directory and save to JSON"""
        all_chunks = []
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(('.txt', '.md', '.pdf')):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    chunks = self.chunk_document(file_path)
                    all_chunks.extend(chunks)
                    print(f"Added {len(chunks)} chunks")
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2)
        
        print(f"Saved {len(all_chunks)} chunks to {output_path}")
        return all_chunks