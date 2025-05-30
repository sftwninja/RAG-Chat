"""
Command line interface handlers for RAG
"""

import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from rag_manager import RAGManager
from chat_interface import ChatInterface

console = Console()


class CLICommands:
    """Handles all CLI commands for the RAG application"""

    def __init__(self, manager: RAGManager):
        self.manager = manager

    # RAG Management Commands
    def handle_rag_create(self, name: str, description: str = ""):
        """Handle RAG creation command"""
        return self.manager.create_rag(name, description)

    def handle_rag_delete(self, name: str, force: bool = False):
        """Handle RAG deletion command"""
        return self.manager.delete_rag(name, force)

    def handle_rag_list(self):
        """Handle RAG list command"""
        self.manager.list_rags()

    def handle_rag_load(self, name: str):
        """Handle RAG load command"""
        return self.manager.load_rag(name)

    def handle_rag_unload(self):
        """Handle RAG unload command"""
        self.manager.unload_rag()

    # Document Management Commands
    def handle_add_document(self, input_source: str, name: Optional[str] = None):
        """Handle document addition command"""
        if not self._check_rag_loaded():
            return False

        if input_source == "-":
            # Read from stdin
            content = sys.stdin.read().strip()
            if content:
                return self.manager.get_current_rag().add_document(content, name or "stdin_input")
            else:
                console.print("[red]No input provided[/red]")
                return False
        else:
            # Add file
            return self.manager.get_current_rag().add_file(input_source)

    def _check_rag_loaded(self) -> bool:
        """Check if a RAG system is currently loaded"""
        # Check if we have a current RAG name set
        if not self.manager.current_rag_name:
            console.print("[red]✗[/red] No RAG system loaded. Use 'rag load <name>' first")
            console.print("Available commands: rag create, rag list, rag load")
            return False
    
        # Try to get/load the current RAG
        # This should trigger _load_current_rag_if_needed if needed
        current_rag = self.manager.get_current_rag()
    
        if not current_rag:
            console.print(f"[red]✗[/red] Failed to load RAG system: {self.manager.current_rag_name}")
            console.print("Available commands: rag create, rag list, rag load")
            return False
    
        return True

    def handle_remove_documents(self, source: str):
        """Handle document removal command"""
        if not self._check_rag_loaded():
            return False

        return self.manager.get_current_rag().remove_documents_by_source(source)

    def handle_list_documents(self):
        """Handle document listing command"""
        if not self._check_rag_loaded():
            return False

        self.manager.get_current_rag().list_documents()
        return True

    def handle_search_documents(self, query: str, count: int = 3):
        """Handle document search command"""
        if not self._check_rag_loaded():
            return False

        results = self.manager.get_current_rag().search_documents(query, count)
        if results:
            console.print(f"[bold]Found {len(results)} relevant documents in '{self.manager.get_current_rag().name}':[/bold]\n")
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('filename', 'Unknown')
                chunk_info = f"chunk {doc.metadata.get('chunk_index', '?')}"
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                console.print(Panel(content, title=f"Result {i}: {source} ({chunk_info})"))
        else:
            console.print("[yellow]No relevant documents found[/yellow]")

        return True

    def handle_query(self, question: str, no_rag: bool = False, show_sources: bool = False):
        """Handle query command"""
        if not self._check_rag_loaded():
            return False

        result = self.manager.get_current_rag().generate_response(question, use_rag=not no_rag)

        rag_indicator = f"RAG ({result['rag_system']})" if result["used_rag"] else f"Direct ({result['rag_system']})"
        console.print(f"\n[bold green]Response ({rag_indicator}):[/bold green]\n{result['answer']}")

        # Show sources if requested and available
        if show_sources and result["source_documents"] and result["used_rag"]:
            console.print("\n[bold]Sources:[/bold]")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get('filename', 'Unknown')
                chunk_info = f"chunk {doc.metadata.get('chunk_index', '?')}"
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                console.print(Panel(preview, title=f"{i}. {source} ({chunk_info})"))

        return True

    def handle_chat(self):
        """Handle interactive chat command"""
        if not self._check_rag_loaded():
            return False

        chat_interface = ChatInterface(self.manager.get_current_rag())
        chat_interface.start_chat()
        return True

    def handle_status(self):
        """Handle status command"""
        current_rag = self.manager.get_current_rag()
        if current_rag:
            doc_count = current_rag._get_document_count()
            qa_available = "Yes" if current_rag.qa_chain else "No"
            console.print(Panel(
                f"[bold]Current RAG System Status[/bold]\n"
                f"Name: {current_rag.name}\n"
                f"Description: {current_rag.description or 'No description'}\n"
                f"Database: {current_rag.db_path}\n"
                f"Documents: {doc_count}\n"
                f"Model: {current_rag.model_name}\n"
                f"Embeddings: {current_rag.embedding_model_name}\n"
                f"QA Chain: {qa_available}",
                title="RAG System Status"
            ))

        # Show all available RAGs
        console.print("\n")
        self.manager.list_rags()
        return True
        
    def handle_model_change(self, model_name: str):
        """Handle model change command"""
        try:
            # Update the model in the manager
            self.manager.set_model(model_name)
            
            # Reload current RAG with new model if one is loaded
            if self.manager.current_rag_name:
                self.manager.unload_rag()
                self.manager.load_rag(self.manager.current_rag_name)
                
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to change model: {e}")
            return False