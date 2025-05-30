"""
RAG Manager for handling multiple RAG systems
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from langchain_ollama import OllamaLLM
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from rag_system import RAGSystem

console = Console()


class RAGManager:
    """Manages multiple RAG systems"""

    def __init__(self, base_path: str = "./rag_systems", model_name: str = "mistral:7b"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.config_file = self.base_path / "config.json"
        self.current_rag: Optional[RAGSystem] = None
        self.available_rags: Dict[str, dict] = {}
        self.current_rag_name: Optional[str] = None
        self.model_name = model_name

        # Load existing RAG configurations
        self._load_config()
        
        # Save the current model to config
        self._save_model_to_config()

    def _load_config(self):
        """Load RAG configurations from disk"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.available_rags = config.get('rags', {})
                self.current_rag_name = config.get('current_rag')
                # Only override model_name from CLI if it's in the config
                saved_model = config.get('model_name')
                if saved_model:
                    self.model_name = saved_model
                console.print(f"[green]✓[/green] Loaded {len(self.available_rags)} RAG configurations")
                console.print(f"[green]✓[/green] Using model: {self.model_name}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
                self.available_rags = {}
                self.current_rag_name = None
        else:
            self.available_rags = {}
            self.current_rag_name = None

    def _load_current_rag_if_needed(self):
        """Load the currently active RAG only when needed (not during list operations)"""
        if self.current_rag is None and self.current_rag_name and self.current_rag_name in self.available_rags:
            try:
                rag_info = self.available_rags[self.current_rag_name]
                
                # Use the RAG's model_name if available, otherwise use the current model_name
                model_to_use = rag_info.get('model_name', self.model_name)
                
                self.current_rag = RAGSystem(
                    name=self.current_rag_name,
                    db_path=rag_info['path'],
                    description=rag_info.get('description', ''),
                    model_name=model_to_use
                )
                # Set up the callback mechanism
                self._setup_rag_callbacks()
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load RAG system: {e}")
                # If loading fails, clear the current RAG reference
                self.current_rag_name = None
                self._save_current_rag(None)

    def _setup_rag_callbacks(self):
        """Set up callback mechanisms for the current RAG"""
        if self.current_rag:
            # Override the notification method to update our config
            self.current_rag._notify_document_count_changed = self._update_document_count

    def _save_current_rag(self, rag_name: Optional[str]):
        """Save the currently loaded RAG name to config"""
        try:
            config = {'rags': self.available_rags}
            config['current_rag'] = rag_name
            self.current_rag_name = rag_name

            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save current RAG state: {e}")

    def _save_config(self):
        """Save RAG configurations to disk"""
        try:
            config = {
                'rags': self.available_rags,
                'current_rag': self.current_rag_name,
                'model_name': self.model_name
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save config: {e}")
            
    def _save_model_to_config(self):
        """Save the current model name to config"""
        # First load existing config to make sure we don't overwrite
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Only update model_name
                config['model_name'] = self.model_name
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                console.print(f"[green]✓[/green] Set active model: {self.model_name}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not update model in config: {e}[/yellow]")
        else:
            self._save_config()
            
    def set_model(self, model_name: str) -> bool:
        """Set the active LLM model for the system"""
        try:
            # Store the old model name in case we need to revert
            old_model_name = self.model_name
            
            # Update model name
            self.model_name = model_name
            
            # Test if model is available
            test_llm = OllamaLLM(model=model_name, temperature=0.7)
            test_response = test_llm.invoke("Hello")
            
            # If successful, save to config
            self._save_model_to_config()
            console.print(f"[green]✓[/green] Changed global model to {model_name}")
            
            return True
        except Exception as e:
            # Revert to previous model on error
            self.model_name = old_model_name if 'old_model_name' in locals() else "mistral:7b"
            console.print(f"[red]✗[/red] Failed to set model {model_name}: {e}")
            console.print("Make sure Ollama is running and model is available:")
            console.print(f"  [bold]ollama serve[/bold]")
            console.print(f"  [bold]ollama pull {model_name}[/bold]")
            return False

    def create_rag(self, name: str, description: str = "") -> bool:
        """Create a new RAG system"""
        if name in self.available_rags:
            console.print(f"[red]✗[/red] RAG '{name}' already exists")
            return False

        try:
            rag_path = self.base_path / name
            rag_path.mkdir(exist_ok=True)

            self.available_rags[name] = {
                'name': name,
                'description': description,
                'path': str(rag_path),
                'created': str(Path().absolute()),  # timestamp could be added
                'document_count': 0,
                'model_name': self.model_name  # Save the current model name with this RAG
            }
    
            self._save_config()
            console.print(f"[green]✓[/green] Created RAG system: {name} with model {self.model_name}")
            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to create RAG: {e}")
            return False

    def delete_rag(self, name: str, force: bool = False) -> bool:
        """Delete a RAG system and optionally its data"""
        if name not in self.available_rags:
            console.print(f"[red]✗[/red] RAG '{name}' not found")
            return False

        try:
            # Unload if currently loaded
            if self.current_rag and self.current_rag.name == name:
                self.unload_rag()

            rag_info = self.available_rags[name]
            rag_path = Path(rag_info['path'])

            # Confirm deletion if not forced
            if not force and rag_path.exists():
                delete_data = Confirm.ask(f"Delete all data for RAG '{name}'?", default=False)
                if delete_data and rag_path.exists():
                    shutil.rmtree(rag_path)
                    console.print(f"[yellow]Deleted data directory for '{name}'[/yellow]")
            elif force and rag_path.exists():
                shutil.rmtree(rag_path)
                console.print(f"[yellow]Deleted data directory for '{name}'[/yellow]")

            # Remove from config
            del self.available_rags[name]
            self._save_config()

            console.print(f"[green]✓[/green] Removed RAG system: {name}")
            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to delete RAG: {e}")
            return False

    def load_rag(self, name: str) -> bool:
        """Load a RAG system"""
        if name not in self.available_rags:
            console.print(f"[red]✗[/red] RAG '{name}' not found")
            return False

        try:
            rag_info = self.available_rags[name]
            
            # Use the RAG's model_name if available, otherwise use the current model_name
            model_to_use = rag_info.get('model_name', self.model_name)
            
            self.current_rag = RAGSystem(
                name=name,
                db_path=rag_info['path'],
                description=rag_info.get('description', ''),
                model_name=model_to_use
            )
    
            # Override the notification method to update our config
            self.current_rag._notify_document_count_changed = self._update_document_count
    
            # Update document count
            doc_count = self.current_rag._get_document_count()
            self.available_rags[name]['document_count'] = doc_count
            self._save_config()
    
            # Save this as the current RAG
            self._save_current_rag(name)
    
            console.print(f"[green]✓[/green] Loaded RAG system: {name} ({doc_count} documents)")
            console.print(f"[green]✓[/green] Using model: {model_to_use}")
            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load RAG: {e}")
            return False

    def unload_rag(self):
        """Unload current RAG system"""
        if self.current_rag_name:
            console.print(f"[yellow]Unloaded RAG system: {self.current_rag_name}[/yellow]")
            self.current_rag = None
            # Clear the current RAG from config
            self._save_current_rag(None)
        else:
            console.print("[yellow]No RAG system currently loaded[/yellow]")

    def list_rags(self):
        """List all available RAG systems"""
        if not self.available_rags:
            console.print("[yellow]No RAG systems available[/yellow]")
            console.print("\nCreate a new RAG system with:")
            console.print(f"  [bold cyan]python main.py {self.model_name} rag create <n>[/bold cyan]")
            return
    
        table = Table(title="Available RAG Systems")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Documents", style="yellow", justify="right")
        table.add_column("Model", style="blue")
        table.add_column("Description", style="dim")
    
        for name, info in self.available_rags.items():
            # Check status based on config only, don't load RAG
            is_loaded = (self.current_rag_name == name)
    
            if is_loaded:
                status = "● LOADED"
                status_style = "bold green"
            else:
                status = "○ Available"
                status_style = "dim"
    
            table.add_row(
                name,
                f"[{status_style}]{status}[/{status_style}]",
                str(info.get('document_count', 0)),
                info.get('model_name', 'mistral:7b'),
                info.get('description', 'No description')
            )
    
        console.print(table)
        console.print(f"\nTotal RAG systems: {len(self.available_rags)}")
        console.print(f"Current default model: [bold blue]{self.model_name}[/bold blue]")
    
        if self.current_rag_name:
            console.print(f"Currently loaded: [bold green]● {self.current_rag_name}[/bold green]")
        else:
            console.print("No RAG system currently loaded")
            console.print("\nLoad a RAG system with:")
            console.print(f"  [bold cyan]python main.py {self.model_name} rag load <n>[/bold cyan]")

    def get_current_rag(self) -> Optional[RAGSystem]:
        """Get the currently loaded RAG system, loading it if necessary"""
        if self.current_rag is None and self.current_rag_name:
            self._load_current_rag_if_needed()
        return self.current_rag
            
    def _update_document_count(self):
        """Update the document count for the current RAG in the config"""
        if self.current_rag and self.current_rag_name:
            doc_count = self.current_rag._get_document_count()
            if self.current_rag_name in self.available_rags:
                self.available_rags[self.current_rag_name]['document_count'] = doc_count
                self._save_config()