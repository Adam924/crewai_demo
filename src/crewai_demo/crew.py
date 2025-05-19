from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# Changed import to avoid deprecation warning
from langchain_openai import OpenAI

# Import RAG tools
from src.crewai_demo.tools.rag_tools import RAGTools

@CrewBase
class CrewaiDemo():
    """CrewaiDemo crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    
    def __init__(self):
        super().__init__()
        # Initialize RAG tools
        self.rag_tools = RAGTools(collection_name="document_chunks")
    
    @agent
    def researcher(self) -> Agent:
        # Create a dictionary-format tool for the retriever
        retrieval_tool = {
            "name": "retrieve_context",
            "description": "Retrieve relevant context for a query",
            "func": self.rag_tools.retrieve_context
        }
        
        return Agent(
            config=self.agents_config['researcher'],  # type: ignore[index]
            tools=[retrieval_tool],  # Added tool in the correct format
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],  # type: ignore[index]
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        config = self.tasks_config['research_task']
        
        # Override context with literal strings instead of using YAML
        context = [
            "Topic: {topic}",
            "Query: {query}",
            "Retrieved Information: {retrieval_context}"
        ]
        
        return Task(
            description=config['description'],
            agent=self.researcher(),
            expected_output=config['expected_output'],
            context=context  # Use our manually defined context
        )

    @task
    def reporting_task(self) -> Task:
        config = self.tasks_config['reporting_task']
        
        # Override context with literal strings instead of using YAML
        context = [
            "Topic: {topic}",
            "Query: {query}",
            "Research: {research_task.output}"
        ]
        
        return Task(
            description=config['description'],
            agent=self.reporting_analyst(),
            expected_output=config['expected_output'],
            context=context,  # Use our manually defined context
            dependencies=[self.research_task()],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiDemo crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )