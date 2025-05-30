"""
Interactive chat interface for RAG systems
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_system import RAGSystem

console = Console()


class ChatInterface:
    """Interactive chat interface for RAG systems"""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.use_rag = True
        self.show_sources = False

    def start_chat(self):
        """Start interactive chat session"""
        console.print(Panel.fit(
            f"[bold blue]RAG Chat Session: {self.rag_system.name}[/bold blue]\n"
            f"Description: {self.rag_system.description or 'No description'}\n"
            f"Documents: {self.rag_system._get_document_count()}\n\n"
            "Type your questions and get RAG-powered responses.\n"
            "Commands: /help, /status, /toggle-rag, /sources, /quit",
            title="Chat Mode"
        ))

        while True:
            try:
                # Get user input
                question = Prompt.ask(f"\n[bold cyan]You ({self.rag_system.name})[/bold cyan]")

                # Handle commands
                command_handled = self._handle_command(question)
                if command_handled:
                    # Special handling for quit commands
                    if question.lower() in ['/quit', '/exit', '/q']:
                        break  # Exit the chat loop
                    continue

                # Generate response
                result = self._generate_response(question)

                # Display response
                self._display_response(result)

            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def _handle_command(self, command: str) -> bool:
        """Handle chat commands. Returns True if command was handled."""
        if command.lower() in ['/quit', '/exit', '/q']:
            console.print("[yellow]Goodbye![/yellow]")
            return True  # Return True to signal handling, but main loop should check for exit condition

        elif command.lower() == '/help':
            console.print(Panel(
                "[bold]Commands:[/bold]\n"
                "/help - Show this help\n"
                "/status - Show system status\n"
                "/toggle-rag - Toggle RAG on/off\n"
                "/sources - Toggle source display\n"
                "/quit - Exit chat"
            ))
            return True

        elif command.lower() == '/status':
            doc_count = self.rag_system._get_document_count()
            rag_status = "ON" if self.use_rag else "OFF"
            sources_status = "ON" if self.show_sources else "OFF"
            console.print(
                f"RAG System: {self.rag_system.name} | Documents: {doc_count} | RAG: {rag_status} | Sources: {sources_status}")
            return True

        elif command.lower() == '/toggle-rag':
            self.use_rag = not self.use_rag
            status = "enabled" if self.use_rag else "disabled"
            console.print(f"[yellow]RAG {status}[/yellow]")
            return True

        elif command.lower() == '/sources':
            self.show_sources = not self.show_sources
            status = "enabled" if self.show_sources else "disabled"
            console.print(f"[yellow]Source display {status}[/yellow]")
            return True

        return False  # Not a command, continue with normal processing

    def _generate_response(self, question: str) -> dict:
        """Generate response with progress indicator"""
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            result = self.rag_system.generate_response(question, self.use_rag)
            progress.update(task, completed=True)

        return result

    def _display_response(self, result: dict):
        """Display the response with optional sources"""
        # Show main response
        rag_indicator = f"RAG: {self.rag_system.name}" if result[
            "used_rag"] else f"Direct"
        console.print(f"\n[bold green]Assistant ({rag_indicator})[/bold green]: {result['answer']}")

        # Show sources if enabled and available
        if self.show_sources and result["source_documents"] and result["used_rag"]:
            console.print("\n[dim]Sources:[/dim]")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get('filename', 'Unknown')
                chunk_info = f"chunk {doc.metadata.get('chunk_index', '?')}"
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                console.print(f"[dim]{i}. {source} ({chunk_info}): {preview}[/dim]")