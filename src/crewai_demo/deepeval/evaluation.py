from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase

class RAGEvaluator:
    """Evaluation of RAG system using DeepEval"""
    
    def __init__(self, threshold=0.7):
        """Initialize the evaluator"""
        self.threshold = threshold
        
        # Initialize metrics
        self.metrics = {
            "contextual_precision": ContextualPrecisionMetric(threshold=threshold),
            "contextual_recall": ContextualRecallMetric(threshold=threshold),
            "contextual_relevancy": ContextualRelevancyMetric(threshold=threshold),
            "answer_relevancy": AnswerRelevancyMetric(threshold=threshold),
            "faithfulness": FaithfulnessMetric(threshold=threshold)
        }
    
    def evaluate(self, query, contexts, response):
        """
        Evaluate a RAG response.
        
        Args:
            query (str): The user query
            contexts (list): Retrieved contexts used for generation
            response (str): The generated response
            
        Returns:
            dict: Evaluation results
        """
        try:
            # Create test case
            context = "\n\n".join(contexts) if isinstance(contexts[0], str) else "\n\n".join([c["text"] for c in contexts])
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                context=context
            )
            
            # Run evaluation
            results = {}
            for name, metric in self.metrics.items():
                metric_result = metric.measure(test_case)
                results[name] = {
                    "score": metric_result.score,
                    "passed": metric_result.passed,
                    "reason": getattr(metric_result, "reason", "No reason provided")
                }
            
            # Calculate overall score
            results["overall"] = {
                "score": sum(r["score"] for name, r in results.items() if name != "overall") / len(self.metrics),
                "passed": all(r["passed"] for name, r in results.items() if name != "overall")
            }
            
            return results
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {"error": str(e)}