"""Example demonstrating batch-llm with LangChain integration.

This example shows how to create custom strategies that integrate with LangChain,
including chains, agents, and RAG (Retrieval-Augmented Generation) pipelines.

Install dependencies:
    pip install 'batch-llm' 'langchain' 'langchain-openai' 'langchain-anthropic' 'langchain-community' 'faiss-cpu'
"""

import asyncio
import os
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import LLMCallStrategy


class LangChainStrategy(LLMCallStrategy[str]):
    """Strategy for using LangChain chains with batch-llm."""

    def __init__(self, chain: LLMChain):
        """
        Initialize LangChain strategy.

        Args:
            chain: Configured LangChain chain
        """
        self.chain = chain

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        """Execute LangChain chain.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        # Run the chain
        result = await self.chain.arun(input=prompt)

        # LangChain doesn't always provide token usage in a standard way
        # For production use, you'd want to extract this from the LLM callbacks
        tokens = {
            "input_tokens": 0,  # Would need callback handler to track
            "output_tokens": 0,
            "total_tokens": 0,
        }

        return result, tokens


# Example 1: Simple LangChain chain with OpenAI
async def example_langchain_openai_chain():
    """Example using LangChain with OpenAI model."""
    print("\n" + "=" * 60)
    print("Example 1: LangChain + OpenAI Chain")
    print("=" * 60 + "\n")

    # Create LangChain LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create prompt template
    template = """You are a helpful assistant that answers questions concisely.

Question: {input}

Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["input"])

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Create strategy
    strategy = LangChainStrategy(chain=chain)

    # Configure processor
    config = ProcessorConfig(max_workers=3, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        questions = [
            "What is the capital of Japan?",
            "Explain photosynthesis briefly.",
            "Who wrote 'Romeo and Juliet'?",
        ]

        for i, question in enumerate(questions):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"question_{i}",
                    strategy=strategy,
                    prompt=question,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nResults:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output}")


# Example 2: LangChain with Anthropic
async def example_langchain_anthropic():
    """Example using LangChain with Anthropic Claude."""
    print("\n" + "=" * 60)
    print("Example 2: LangChain + Anthropic Claude")
    print("=" * 60 + "\n")

    # Create LangChain LLM
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=1.0,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Create prompt template for summarization
    template = """Please summarize the following text in 2-3 sentences:

{input}

Summary:"""

    prompt = PromptTemplate(template=template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)

    # Create strategy
    strategy = LangChainStrategy(chain=chain)

    # Configure processor
    config = ProcessorConfig(max_workers=2, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        documents = [
            """
            Artificial intelligence (AI) is transforming healthcare through improved
            diagnostics, personalized treatment plans, and drug discovery. Machine
            learning algorithms can analyze medical images with high accuracy, often
            detecting patterns that human radiologists might miss. AI is also being
            used to predict patient outcomes and optimize hospital operations.
            """,
            """
            Renewable energy sources like solar and wind power are becoming increasingly
            cost-competitive with fossil fuels. The technology has improved dramatically
            over the past decade, with solar panel efficiency increasing and costs
            decreasing. Many countries are now investing heavily in renewable
            infrastructure to meet climate goals.
            """,
        ]

        for i, doc in enumerate(documents):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"doc_{i}",
                    strategy=strategy,
                    prompt=doc,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nSummaries:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output}")


# Example 3: RAG with LangChain and FAISS
async def example_langchain_rag():
    """Example using LangChain RAG pipeline with batch processing."""
    print("\n" + "=" * 60)
    print("Example 3: LangChain RAG Pipeline")
    print("=" * 60 + "\n")

    from langchain.chains import RetrievalQA
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    class RAGStrategy(LLMCallStrategy[str]):
        """Custom strategy for RAG with LangChain."""

        def __init__(self, qa_chain: RetrievalQA):
            self.qa_chain = qa_chain

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[str, dict[str, int]]:
            # Run the RAG chain
            result = await self.qa_chain.arun(prompt)

            tokens = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

            return result, tokens

    # Sample documents for our knowledge base
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural networks are inspired by the structure of the human brain.",
        "Natural language processing (NLP) enables computers to understand human language.",
        "Deep learning uses multiple layers of neural networks for complex pattern recognition.",
    ]

    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    # Create strategy
    strategy = RAGStrategy(qa_chain=qa_chain)

    # Configure processor
    config = ProcessorConfig(max_workers=2, timeout_per_item=30.0)

    # Process questions
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        questions = [
            "What is Python?",
            "How do neural networks work?",
            "What is the relationship between AI and machine learning?",
        ]

        for i, question in enumerate(questions):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"rag_question_{i}",
                    strategy=strategy,
                    prompt=question,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nRAG Answers:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output}")


# Example 4: Different chains for different item types
async def example_langchain_multi_chain():
    """Example using different LangChain chains for different task types."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple LangChain Chains")
    print("=" * 60 + "\n")

    # Create different LLMs and chains
    openai_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    anthropic_llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=1.0,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Chain for factual questions
    fact_template = """Answer this factual question concisely:

{input}

Answer:"""
    fact_prompt = PromptTemplate(template=fact_template, input_variables=["input"])
    fact_chain = LLMChain(llm=openai_llm, prompt=fact_prompt)

    # Chain for creative tasks
    creative_template = """Write a creative response to this prompt:

{input}

Creative response:"""
    creative_prompt = PromptTemplate(
        template=creative_template, input_variables=["input"]
    )
    creative_chain = LLMChain(llm=anthropic_llm, prompt=creative_prompt)

    # Create strategies
    fact_strategy = LangChainStrategy(chain=fact_chain)
    creative_strategy = LangChainStrategy(chain=creative_chain)

    # Configure processor
    config = ProcessorConfig(max_workers=4, timeout_per_item=30.0)

    # Process mixed task types
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        # Factual questions
        await processor.add_work(
            LLMWorkItem(
                item_id="fact_1",
                strategy=fact_strategy,
                prompt="What is the speed of light?",
            )
        )
        await processor.add_work(
            LLMWorkItem(
                item_id="fact_2",
                strategy=fact_strategy,
                prompt="When was the Declaration of Independence signed?",
            )
        )

        # Creative tasks
        await processor.add_work(
            LLMWorkItem(
                item_id="creative_1",
                strategy=creative_strategy,
                prompt="Write a haiku about artificial intelligence.",
            )
        )
        await processor.add_work(
            LLMWorkItem(
                item_id="creative_2",
                strategy=creative_strategy,
                prompt="Describe a futuristic city in one sentence.",
            )
        )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nResults by type:")

    for item_result in result.results:
        if item_result.success:
            task_type = "FACT" if item_result.item_id.startswith("fact") else "CREATIVE"
            print(f"\n[{task_type}] {item_result.item_id}:")
            print(f"  {item_result.output}")


async def main():
    """Run all examples."""
    # Check for required API keys
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("Error: At least one API key must be set:")
        print("  - OPENAI_API_KEY for OpenAI examples")
        print("  - ANTHROPIC_API_KEY for Anthropic examples")
        return

    # Run examples based on available API keys
    if has_openai:
        await example_langchain_openai_chain()
        await example_langchain_rag()

    if has_anthropic:
        await example_langchain_anthropic()

    if has_openai and has_anthropic:
        await example_langchain_multi_chain()


if __name__ == "__main__":
    asyncio.run(main())
