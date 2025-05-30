# Multi-RAG CLI Application

A command-line application for managing multiple RAG (Retrieval-Augmented Generation) systems using Ollama with customizable models and LangChain.

## Features

- **Multiple RAG Systems**: Create, manage, and switch between different RAG systems for different domains or projects
- **Model Selection**: Choose different Ollama models for each RAG system or globally
- **Document Management**: Add, remove, and search documents within each RAG system
- **Interactive Chat**: Chat against your documents using natural language
- **Persistent Storage**: All RAG systems and documents are saved to disk
- **Rich CLI Interface**: Nice-looking terminal interface with colors, tables, and progress indicators
- **Source Attribution**: Track and display sources for generated responses

## Project Structure

```
rag_cli/
├── main.py              # Main CLI entry point
├── rag_manager.py       # Manages multiple RAG systems
├── rag_system.py        # Core RAG system implementation
├── chat_interface.py    # Interactive chat interface
├── cli_commands.py      # Command handlers
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Ollama:**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Pull Mistral model
   ollama pull mistral:7b
   ```

3. **Make the script executable:**
   ```bash
   chmod +x main.py
   ```

## Usage

### RAG System Management

**Get help for RAG commands:**
```bash
python main.py rag
# Shows all available RAG subcommands
```

**Create RAG systems:**
```bash
# Create different RAG systems for different topics
python main.py rag create "machine_learning" --description "AI and ML documentation"
python main.py rag create "company_docs" --description "Internal company documentation"
python main.py rag create "research_papers" --description "Academic research collection"
python main.py rag create "character_sheet" --description "DnD character sheet"
```

**List all RAG systems:**
```bash
python main.py rag list
# Shows all RAG systems with current load status
```

**Load a specific RAG system:**
```bash
python main.py rag load machine_learning
```

**Switch between RAG systems:**
```bash
python main.py rag unload
python main.py rag load company_docs
```

**Delete a RAG system:**
```bash
# Delete with confirmation prompt
python main.py rag delete old_rag

# Force delete without prompts
python main.py rag delete old_rag --force
```

### Document Management

**Add documents (requires loaded RAG):**
```bash
# Load a RAG first
python main.py rag load machine_learning

# Add files
python main.py add document.txt

# Add from stdin
echo "This is my knowledge" | python main.py add - --name "my_notes"
```

**Remove documents:**
```bash
python main.py remove document.txt
```

**List documents:**
```bash
python main.py list
```

**Search documents:**
```bash
python main.py search "machine learning algorithms" --count 5
```

### Querying and Chat

**Ask questions:**
```bash
# Query with RAG
python main.py query "What is deep learning?"

# Query without RAG (direct LLM)
python main.py query "What is deep learning?" --no-rag

# Show sources
python main.py query "What is deep learning?" --show-sources
```

**Interactive chat:**
```bash
python main.py chat
```

**Chat commands:**
- `/help` - Show available commands
- `/status` - Show system status
- `/toggle-rag` - Toggle RAG on/off
- `/sources` - Toggle source display
- `/quit` - Exit chat

**Check status:**
```bash
python main.py status
```

## Example Workflow

```bash
# 1. Create specialized RAG systems
python main.py rag create "tech_docs" --description "Technical documentation"
python main.py rag create "business_docs" --description "Business and strategy docs"

# 2. Load and populate first RAG
python main.py rag load tech_docs
python main.py add api_documentation.md
python main.py add coding_standards.txt

# 3. Switch to business RAG
python main.py rag load business_docs
python main.py add quarterly_report.txt
python main.py add strategy_doc.txt

# 4. Chat with business context
python main.py chat

# 5. Check status of all systems
python main.py status
```

## Configuration

I tried to stick with sensible defaults, but you can modify `config.py` to customize:

- Model settings (LLM and embedding models)
- Text processing parameters (chunk size, overlap)
- File type handling
- Retrieval settings

## Architecture

- **RAGManager**: Manages multiple RAG systems, handles creation/deletion/loading
- **RAGSystem**: Core implementation of individual RAG systems using LangChain
- **ChatInterface**: Handles interactive chat sessions
- **CLICommands**: Processes all command-line operations
- **Config**: Centralized configuration management

## Benefits

- **Organization**: Separate knowledge bases for different domains
- **Memory Efficiency**: Only load the RAG system you're currently using
- **Flexibility**: Switch contexts easily between different projects
- **Maintainability**: Clean separation of concerns across multiple files
- **Extensibility**: Easy to add new features or modify existing ones

## Requirements

- Python 3.8+
- Ollama
- Dependencies listed in requirements.txt

## Troubleshooting

**Ollama connection issues:**
- Ensure Ollama is running: `ollama serve`
- Verify model is available: `ollama list`
- Pull model if needed: `ollama pull mistral:7b`

**Import errors:**
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version compatibility

**Memory issues:**
- Use smaller chunk sizes in config.py
- Reduce the number of retrieved documents (search_k)
- Unload RAG systems when not in use

## Wrapper Scripts

The application includes wrapper scripts that allow you to run commands without typing `python main.py` every time:

### Linux/macOS (Bash)

1. **Make the script executable:**
   ```bash
   chmod +x ragcli.sh
   ```

2. **Run commands using the script:**
   ```bash
   ./ragcli.sh query "What is deep learning?"
   ./ragcli.sh rag list
   ./ragcli.sh chat
   ```

### Windows Command Prompt (CMD)

1. **Run commands using the batch file:**
   ```cmd
   ragcli.bat rag list
   ragcli.bat query "What is deep learning?"
   ragcli.bat chat
   ```

2. **Add to PATH (optional):**
   To run the `ragcli.bat` command from any directory:
   - Add the directory containing `ragcli.bat` to your system PATH, or
   - Create a shortcut to `ragcli.bat` in a directory that's already in your PATH

3. **Create command shortcuts (optional):**
   You can create custom command shortcuts by creating additional batch files:
   ```cmd
   @echo off
   ragcli.bat chat %*
   ```
   Save as `ragchat.bat` to quickly start chat sessions with: `ragchat`
