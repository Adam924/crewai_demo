from src.crewai_demo.tools.vectordb import VectorDB

class RAGTools:
    """Tools for Retrieval-Augmented Generation"""
    
    def __init__(self, collection_name="document_chunks"):
        """Initialize the RAG tools"""
        self.vector_db = VectorDB(collection_name=collection_name)
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context for a query"""
        results = self.vector_db.search(query, top_k=top_k)
        
        if not results:
            return "No relevant information found."
        
        # Format as context string
        context = "\n\n".join([f"Context #{i+1}:\n{result['text']}" for i, result in enumerate(results)])
        return context
    
    def get_relevant_chunks(self, query, top_k=3):
        """Get relevant chunks for a query"""
        return self.vector_db.search(query, top_k=top_k)