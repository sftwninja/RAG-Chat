#!/usr/bin/env python3
"""
Multi-RAG CLI Application for Ollama using LangChain
Main entry point for the command-line interface
"""

import argparse

from rag_manager import RAGManager
from cli_commands import CLICommands


def create_parser():
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(description="Multi-RAG CLI for Ollama LLMs with LangChain")
    parser.add_argument("--base-path", default="./rag_systems", help="Base path for RAG systems")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # RAG management commands
    rag_parser = subparsers.add_parser("rag", help="RAG system management")
    rag_subparsers = rag_parser.add_subparsers(dest="rag_command", help="RAG commands")

    # Create RAG
    create_rag_parser = rag_subparsers.add_parser("create", help="Create new RAG system")
    create_rag_parser.add_argument("name", help="RAG system name")
    create_rag_parser.add_argument("--description", help="RAG system description")

    # Delete RAG
    delete_rag_parser = rag_subparsers.add_parser("delete", help="Delete RAG system")
    delete_rag_parser.add_argument("name", help="RAG system name")
    delete_rag_parser.add_argument("--force", action="store_true", help="Force delete without confirmation")

    # List RAGs
    rag_subparsers.add_parser("list", help="List all RAG systems")

    # Load RAG
    load_rag_parser = rag_subparsers.add_parser("load", help="Load RAG system")
    load_rag_parser.add_argument("name", help="RAG system name")

    # Unload RAG
    rag_subparsers.add_parser("unload", help="Unload current RAG system")

    # Document management commands (require loaded RAG)
    add_parser = subparsers.add_parser("add", help="Add document or file")
    add_parser.add_argument("input", help="File path or '-' for stdin")
    add_parser.add_argument("--name", help="Document name (for stdin input)")

    remove_parser = subparsers.add_parser("remove", help="Remove documents by source")
    remove_parser.add_argument("source", help="Source name/path to remove")

    subparsers.add_parser("list", help="List all documents in current RAG")

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--count", type=int, default=3, help="Number of results")

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    query_parser.add_argument("--show-sources", action="store_true", help="Show source documents")

    subparsers.add_parser("chat", help="Start interactive chat")
    subparsers.add_parser("status", help="Show system status")

    # Model command
    model_parser = subparsers.add_parser("model", help="Change the active LLM model")
    model_parser.add_argument("model_name", help="Name of the model to use (e.g., mistral:7b, llama3.2:3b)")
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Initialize RAG manager and CLI commands
    manager = RAGManager(base_path=args.base_path)
    cli = CLICommands(manager)

    # Handle RAG management commands
    if args.command == "rag":
        if args.rag_command == "create":
            cli.handle_rag_create(args.name, args.description or "")
        elif args.rag_command == "delete":
            cli.handle_rag_delete(args.name, args.force)
        elif args.rag_command == "list":
            cli.handle_rag_list()
        elif args.rag_command == "load":
            cli.handle_rag_load(args.name)
        elif args.rag_command == "unload":
            cli.handle_rag_unload()
        elif args.rag_command is None:
            # Show help when no subcommand is provided
            print("usage: main.py rag [-h] {create,delete,list,load,unload} ...")
            print("")
            print("RAG system management")
            print("")
            print("positional arguments:")
            print("  {create,delete,list,load,unload}")
            print("                        RAG commands")
            print("    create              Create new RAG system")
            print("    delete              Delete RAG system")
            print("    list                List all RAG systems")
            print("    load                Load RAG system")
            print("    unload              Unload current RAG system")
            print("")
            print("optional arguments:")
            print("  -h, --help            show this help message and exit")
        else:
            # Show help for unknown subcommand
            print("usage: main.py rag [-h] {create,delete,list,load,unload} ...")
            print("")
            print("RAG system management")
            print("")
            print("positional arguments:")
            print("  {create,delete,list,load,unload}")
            print("                        RAG commands")
            print("    create              Create new RAG system")
            print("    delete              Delete RAG system")
            print("    list                List all RAG systems")
            print("    load                Load RAG system")
            print("    unload              Unload current RAG system")
            print("")
            print("optional arguments:")
            print("  -h, --help            show this help message and exit")
        return

    # Handle document management and query commands
    if args.command == "add":
        cli.handle_add_document(args.input, args.name)
    elif args.command == "remove":
        cli.handle_remove_documents(args.source)
    elif args.command == "list":
        cli.handle_list_documents()
    elif args.command == "search":
        cli.handle_search_documents(args.query, args.count)
    elif args.command == "query":
        cli.handle_query(args.question, args.no_rag, args.show_sources)
    elif args.command == "chat":
        cli.handle_chat()
    elif args.command == "status":
        cli.handle_status()
    elif args.command == "model":
        cli.handle_model_change(args.model_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()