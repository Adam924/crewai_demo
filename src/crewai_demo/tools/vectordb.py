from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

class VectorDB:
    """Vector database implementation using Qdrant"""
    
    def __init__(self, collection_name="document_chunks", model_name="all-MiniLM-L6-v2"):
        """Initialize the vector database with Qdrant"""
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created collection {self.collection_name}")
        except Exception as e:
            print(f"Error checking/creating collection: {e}")
    
    def store_chunks(self, chunks):
        """Store chunks in the vector database"""
        if not chunks:
            return 0
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    }
                )
            )
        
        # Store in Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Stored {len(points)} chunks in Qdrant")
        return len(points)
    
    def search(self, query, top_k=3):
        """Search for relevant chunks"""
        # Embed the query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in Qdrant
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result.payload["text"],
                "metadata": result.payload["metadata"],
                "score": result.score
            })
        
        return formatted_results