"""
Core RAG System implementation using LangChain
"""

import os
from datetime import datetime
import hashlib
from pathlib import Path
from typing import List

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class RAGSystem:
    """Individual RAG system for managing documents and generating responses"""

    def __init__(self, name: str, db_path: str, description: str = "", model_name: str = "mistral:7b"):
        self.name = name
        self.description = description
        self.db_path = db_path
        self.model_name = model_name
        self.embedding_model_name = "all-MiniLM-L6-v2"

        # Initialize components
        self._init_embeddings()
        self._init_vectorstore()
        self._init_llm()
        self._init_qa_chain()

    def _init_embeddings(self):
        """Initialize sentence transformer embeddings"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading embedding model...", total=None)
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name
                )
                progress.update(task, completed=True)
            console.print(f"[green]✓[/green] Embedding model loaded: {self.embedding_model_name}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load embedding model: {e}")
            raise

    def _init_vectorstore(self):
        """Initialize ChromaDB vector store"""
        try:
            # Check if database exists
            if os.path.exists(self.db_path) and os.listdir(self.db_path):
                # Load existing database
                self.vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
                console.print(f"[green]✓[/green] Loaded existing database with {self._get_document_count()} documents")
            else:
                # Create empty database
                self.vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
                console.print(f"[green]✓[/green] Created new database at {self.db_path}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize database: {e}")
            raise

    def _init_llm(self):
        """Initialize Ollama LLM"""
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=0.7,
            )

            # Test the connection using invoke method
            test_response = self.llm.invoke("Hello")
            console.print(f"[green]✓[/green] Ollama {self.model_name} is ready")
        except Exception as e:
            console.print(f"[red]✗[/red] Ollama connection failed: {e}")
            console.print("Make sure Ollama is running and model is available:")
            console.print("  [bold]ollama serve[/bold]")
            console.print(f"  [bold]ollama pull {self.model_name}[/bold]")
            raise

    def _init_qa_chain(self):
        """Initialize the QA chain"""
        # Custom prompt template
        current_date = datetime.now().date()
        prompt_template = f"""Today's date is {current_date.strftime('%Y-%m-%d')}. You are an AI assistant for the "{self.name}" knowledge base. Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{{context}}

Question: {{question}}

Answer:"""

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Initialize QA chain (will be updated when documents are available)
        self._update_qa_chain()

    def _update_qa_chain(self):
        """Update the QA chain based on current vectorstore state"""
        try:
            doc_count = self._get_document_count()
            if doc_count > 0:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": 3}  # Return top 3 relevant documents
                    ),
                    chain_type_kwargs={"prompt": self.prompt},
                    return_source_documents=True
                )
            else:
                self.qa_chain = None
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create QA chain: {e}[/yellow]")
            self.qa_chain = None

    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash of the document content for duplicate detection"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _check_document_exists(self, content_hash: str, filepath: str = None) -> dict:
        """
        Check if a document with the same content or filepath already exists
        Returns dict with 'exists' boolean, 'doc_id' if matched by filepath, and relevant ids to remove
        """
        result = {'exists': False, 'doc_id': None, 'match_type': None, 'ids_to_remove': []}
        try:
            # Get all documents with metadata (ids come by default)
            all_docs = self.vectorstore.get(include=['metadatas'])

            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata:
                    # Check by content hash
                    if metadata.get('content_hash') == content_hash:
                        result['exists'] = True
                        result['match_type'] = 'content'
                        result['ids_to_remove'].append(all_docs['ids'][i])

                    # Check by filepath (if provided)
                    if filepath and metadata.get('filepath') == str(Path(filepath).absolute()):
                        result['exists'] = True
                        result['match_type'] = 'filepath'
                        result['doc_id'] = all_docs['ids'][i]
                        result['ids_to_remove'].append(all_docs['ids'][i])

            return result
        except Exception as e:
            console.print(f"[yellow]Warning: Error checking document existence: {e}[/yellow]")
            return result

    def _get_document_count(self) -> int:
        """Get the number of documents in the vectorstore"""
        try:
            return len(self.vectorstore.get()['ids'])
        except:
            return 0

    def _split_text(self, text: str, filename: str = "") -> List[Document]:
        """Split text into chunks using LangChain text splitter"""
        # Choose splitter based on content type
        if filename.endswith('.py') or filename.endswith('.js') or filename.endswith('.html'):
            # Use recursive splitter for code
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            # Use character splitter for regular text
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            )

        # Create document
        doc = Document(
            page_content=text,
            metadata={
                "filename": filename,
                "content_length": len(text),
                "rag_system": self.name,
                "content_hash": self._generate_content_hash(text)
            }
        )

        # Split into chunks
        chunks = text_splitter.split_documents([doc])

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["rag_system"] = self.name

        return chunks

    def add_document(self, content: str, filename: str = "", metadata: dict = None) -> bool:
        """Add a document to the vector database, replacing any existing versions"""
        try:
            # Check for duplicates
            content_hash = self._generate_content_hash(content)
            doc_check = self._check_document_exists(content_hash)

            # If document exists, remove the old version
            if doc_check['exists'] and doc_check['ids_to_remove']:
                self.vectorstore.delete(doc_check['ids_to_remove'])
                console.print(f"[yellow]⚠[/yellow] Replacing existing document in '{self.name}': {filename or 'text'}")

            # Split document into chunks
            chunks = self._split_text(content, filename)

            if not chunks:
                console.print("[yellow]⚠[/yellow] No content to add")
                return False

            # Add additional metadata
            for chunk in chunks:
                if metadata:
                    chunk.metadata.update(metadata)

            # Add to vectorstore
            self.vectorstore.add_documents(chunks)

            # Update QA chain
            self._update_qa_chain()

            if doc_check['exists']:
                console.print(
                    f"[green]✓[/green] Updated document in '{self.name}': {filename or 'text'} ({len(chunks)} chunks)")
            else:
                console.print(
                    f"[green]✓[/green] Added document to '{self.name}': {filename or 'text'} ({len(chunks)} chunks)")

            # Notify parent that document count may have changed
            self._notify_document_count_changed()

            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to add document: {e}")
            return False

    def add_file(self, filepath: str) -> bool:
        """Add a file to the vector database, replacing any existing versions"""
        try:
            path = Path(filepath)
            if not path.exists():
                console.print(f"[red]✗[/red] File not found: {filepath}")
                return False

            # Use LangChain's TextLoader
            try:
                loader = TextLoader(str(path), encoding='utf-8')
                documents = loader.load()
            except Exception as e:
                console.print(f"[red]✗[/red] Cannot read file: {e}")
                return False

            if not documents:
                console.print(f"[yellow]⚠[/yellow] No content loaded from file")
                return False

            # Check for duplicates using file path and content
            content = documents[0].page_content
            content_hash = self._generate_content_hash(content)

            doc_check = self._check_document_exists(content_hash, filepath)

            # If file exists, remove the old version
            if doc_check['exists'] and doc_check['ids_to_remove']:
                self.vectorstore.delete(doc_check['ids_to_remove'])
                console.print(f"[yellow]⚠[/yellow] Replacing existing file in '{self.name}': {path.name}")

            # Process each document
            all_chunks = []
            for doc in documents:
                chunks = self._split_text(doc.page_content, path.name)
                for chunk in chunks:
                    chunk.metadata.update({
                        "filepath": str(path.absolute()),
                        "source": str(path)
                    })
                all_chunks.extend(chunks)

            # Add to vectorstore
            self.vectorstore.add_documents(all_chunks)

            # Update QA chain
            self._update_qa_chain()

            if doc_check['exists']:
                console.print(f"[green]✓[/green] Updated file in '{self.name}': {path.name} ({len(all_chunks)} chunks)")
            else:
                console.print(f"[green]✓[/green] Added file to '{self.name}': {path.name} ({len(all_chunks)} chunks)")

            # Notify parent that document count may have changed
            self._notify_document_count_changed()

            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to add file: {e}")
            return False

    def remove_documents_by_source(self, source_filter: str) -> bool:
        """Remove documents by source (filename or filepath)"""
        try:
            # Get all documents
            all_docs = self.vectorstore.get(include=['metadatas'])

            if not all_docs['ids']:
                console.print("[yellow]No documents to remove[/yellow]")
                return False

            # Find matching documents
            ids_to_remove = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata and (
                    metadata.get('filename', '') == source_filter or
                    metadata.get('source', '') == source_filter or
                    source_filter in metadata.get('filepath', '')
                ):
                    ids_to_remove.append(all_docs['ids'][i])

            if not ids_to_remove:
                console.print(f"[yellow]No documents found matching: {source_filter}[/yellow]")
                return False

            # Remove documents
            self.vectorstore.delete(ids_to_remove)

            # Update QA chain
            self._update_qa_chain()

            console.print(f"[green]✓[/green] Removed {len(ids_to_remove)} document chunks from '{self.name}' matching: {source_filter}")

            # Notify parent that document count may have changed
            self._notify_document_count_changed()

            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to remove documents: {e}")
            return False

    def list_documents(self):
        """List all documents in the database"""
        try:
            # Get all documents with metadata
            all_docs = self.vectorstore.get(include=['metadatas'])

            if not all_docs['ids']:
                console.print(f"[yellow]No documents in RAG system '{self.name}'[/yellow]")
                return

            # Group by source/filename
            sources = {}
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata:
                    source = metadata.get('filename') or metadata.get('source', f"doc_{i}")
                    if source not in sources:
                        sources[source] = {
                            'chunks': 0,
                            'total_length': 0,
                            'filepath': metadata.get('filepath', 'N/A')
                        }
                    sources[source]['chunks'] += 1
                    sources[source]['total_length'] += metadata.get('content_length', 0)

            # Create table
            table = Table(title=f"Documents in RAG System: {self.name}")
            table.add_column("Source", style="cyan")
            table.add_column("Chunks", style="yellow", justify="right")
            table.add_column("Total Length", style="green", justify="right")
            table.add_column("Path", style="dim")

            for source, info in sources.items():
                table.add_row(
                    source,
                    str(info['chunks']),
                    str(info['total_length']),
                    info['filepath']
                )

            console.print(table)
            console.print(f"\nTotal chunks: {len(all_docs['ids'])}")
            console.print(f"Total sources: {len(sources)}")

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to list documents: {e}")

    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents"""
        try:
            if self._get_document_count() == 0:
                console.print(f"[yellow]No documents in RAG system '{self.name}' to search[/yellow]")
                return []

            # Use vectorstore similarity search
            results = self.vectorstore.similarity_search(query, k=k)
            return results

        except Exception as e:
            console.print(f"[red]✗[/red] Search failed: {e}")
            return []

    def generate_response(self, question: str, use_rag: bool = True) -> dict:
        """Generate response using LangChain with optional RAG"""
        try:
            if use_rag and self.qa_chain and self._get_document_count() > 0:
                # Use RAG with QA chain
                # Replace deprecated __call__ method with invoke method
                result = self.qa_chain.invoke({"query": question})
                return {
                    "answer": result["result"],
                    "source_documents": result.get("source_documents", []),
                    "used_rag": True,
                    "rag_system": self.name
                }
            else:
                # Direct LLM response using invoke method
                response = self.llm.invoke(question)
                return {
                    "answer": response,
                    "source_documents": [],
                    "used_rag": False,
                    "rag_system": self.name
                }

        except Exception as e:
            return {
                "answer": f"Error generating response: {e}",
                "source_documents": [],
                "used_rag": False,
                "rag_system": self.name
            }

    def _notify_document_count_changed(self):
        """Notify that the document count has changed (to be overridden by parent)"""
        # This method will be overridden by the RAGManager to update config
        pass