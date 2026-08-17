# Multi-Agent AI System with Supervisor

A multi-agent AI application built with LangGraph that routes user requests to specialized agents for conversation, database querying, web research, document retrieval, and data visualization.

The system uses a supervisor architecture to determine which agent should handle each request and supports multi-step workflows, such as retrieving data from a PostgreSQL database and passing the results to the visualization agent.

## Features

- Supervisor-based request routing
- General conversation agent
- PostgreSQL SQL agent
- Web research agent
- Retrieval-Augmented Generation (RAG) agent
- Data visualization agent
- SQL-to-visualization agent coordination
- Conversation memory using LangGraph checkpoints
- Error handling and validation for unsupported or unavailable data
- LangSmith tracing and observability

## Architecture

The application uses a LangGraph supervisor architecture.

The supervisor analyzes each user request and routes it to one of five specialized agents:

- **Conversation Agent** – Handles greetings, explanations, and general conversation.
- **SQL Agent** – Generates safe PostgreSQL `SELECT` queries, executes them against the database, and converts the results into natural-language answers.
- **Web Research Agent** – Searches the web for current information and generates answers grounded in the retrieved search results.
- **RAG Agent** – Retrieves relevant information from internal company documents using FAISS and sentence-transformer embeddings.
- **Visualization Agent** – Creates bar and pie charts from user-provided data or results produced by the SQL agent.

For database visualization requests, the workflow can involve multiple agents:

```text
User → Supervisor → SQL Agent → Visualization Agent → End
```

Other requests are routed directly to the appropriate specialized agent.

## Technologies Used

- Python
- LangChain
- LangGraph
- Groq
- GPT-OSS 120B
- LangSmith
- PostgreSQL
- psycopg2
- FAISS
- Hugging Face Sentence Transformers
- Serper API
- Matplotlib
- python-dotenv

## Project Structure

```text
AI-Project1/
│
├── main.py
├── graph.py
├── nodes.py
├── state.py
├── llm.py
├── database.py
├── web_search.py
├── visualization.py
│
├── rag/
│   ├── retriever.py
│   ├── documents/
│   │   ├── employee_handbook.txt
│   │   └── remote_work_policy.txt
│   └── vector_store/
│
├── .env
├── .gitignore
└── README.md
```

## Agents

### Supervisor Agent

Acts as the entry point of the graph. It classifies each request into one of the following categories:

- `conversation`
- `sql`
- `web_research`
- `visualization`
- `rag`

It also determines whether a database request requires visualization, allowing SQL results to be passed to the visualization agent.

### Conversation Agent

Handles requests that do not require access to specialized tools or data sources.

**Examples:**

```text
Hello, how are you?
Explain what a relational database is.
```

### SQL Agent

Converts natural-language questions into PostgreSQL queries.

The SQL agent:

- Uses the live database schema when generating queries.
- Generates `SELECT` queries only.
- Uses a read-only database connection.
- Converts database results into natural-language responses.
- Handles missing or nonexistent database information.
- Preserves numeric database values for downstream visualization.

**Example:**

```text
Which customer spent the most money?
```

### Web Research Agent

Handles questions requiring current external information.

Search results are retrieved using the **Serper API** and supplied to the language model as context. The agent is instructed to base its answer only on the retrieved results and provide source information.

**Example:**

```text
What is the latest stable PostgreSQL version?
```

### RAG Agent

Answers questions using internal company documents.

The RAG pipeline uses:

- **TextLoader** to load documents.
- **RecursiveCharacterTextSplitter** to divide them into chunks.
- **sentence-transformers/all-MiniLM-L6-v2** to generate embeddings.
- **FAISS** for vector similarity search.
- A relevance threshold to reject unrelated retrieved documents.

If the requested information is not supported by the internal documents, the agent reports that the information is unavailable instead of generating an unsupported answer.

### Visualization Agent

Creates visualizations using **Matplotlib**.

Currently supported chart types:

- Bar charts
- Pie charts

The agent supports directly provided data:

```text
Create a pie chart from A=40, B=35, C=25.
```

It also supports database-backed visualization:

```text
Create a chart showing total spending by customer.
```

For database-backed charts, the supervisor first routes the request to the SQL agent. The resulting structured data is then passed to the visualization agent.

## Conversation Memory

The graph uses LangGraph's `InMemorySaver` checkpointer.

Messages using the same `thread_id` share conversation state, allowing the assistant to maintain context across multiple turns during the running session.

The current implementation uses in-memory rather than persistent storage, so conversation state is reset when the application process is restarted.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Python environment

Create and activate a virtual environment or Conda environment.

Using Conda:

```bash
conda create -n ai-project python=3.14
conda activate ai-project
pip install -r requirements.txt
```

### 3. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key

DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=5432
```

The `.env` file should **not** be committed to GitHub.

### 5. Configure the PostgreSQL database

The SQL agent expects a PostgreSQL database containing the project database schema.

The current project uses the following tables:

- `customers`
- `suppliers`
- `products`
- `orders`
- `order_items`

### 6. Build the RAG vector store

Place the internal documents inside:

```text
rag/documents/
```

Run the vector-store creation script to generate the FAISS index before using the RAG agent.

## Running the Application

Run:

```bash
python main.py
```

The application starts an interactive conversation:

```text
AI Assistant
Type 'exit' to end the conversation.

YOU: How many annual leave days do employees receive?
ASSISTANT: Full-time employees receive 20 paid annual leave days per calendar year.

YOU: Which customer spent the most money?
ASSISTANT: ...
```

Enter `exit` or `quit` to end the session.

## Safety and Reliability

Several safeguards are included in the system:

- PostgreSQL connections are configured as read-only.
- The SQL agent is restricted to `SELECT` queries.
- Generated SQL is validated before execution.
- Numeric SQL results remain numeric for visualization.
- RAG retrieval uses a relevance threshold to reject unrelated documents.
- Web answers are constrained to retrieved search results.
- Visualization data is validated before chart generation.
- Missing or unsupported information produces explicit fallback responses rather than fabricated data.
