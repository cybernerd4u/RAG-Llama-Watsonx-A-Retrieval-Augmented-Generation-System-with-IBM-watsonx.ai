# RAG-Llama-Watsonx: A Retrieval-Augmented Generation System with IBM watsonx.ai

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system using **IBM watsonx.ai** and the **Llama 3 405B Instruct** model. It fetches web data from Wikipedia pages, processes it into chunks, stores embeddings in a Chroma vector store, and uses the Llama model to generate concise answers to user queries. The system is designed for question-answering tasks, leveraging web content from IBM Watsonx and IBM Cloud Wikipedia pages to provide accurate, context-aware responses.

### Demo
[![Watch a demo of the project on YouTube](https://img.youtube.com/vi/TfA2m1f25Jk/0.jpg)](https://www.youtube.com/watch?v=TfA2m1f25Jk)

## Features
- **Web Data Retrieval**: Uses `WebBaseLoader` to fetch content from specified URLs.
- **Text Processing**: Splits documents into chunks using `RecursiveCharacterTextSplitter` for efficient retrieval.
- **Vector Store**: Embeds documents with IBM’s `slate-125m-english-rtrvr` model and stores them in a Chroma vector database.
- **RAG Pipeline**: Combines retrieval from the vector store with Llama 3 405B for generating answers.
- **Customizable Queries**: Supports a variety of questions about IBM, watsonx, and IBM Cloud.
- **Environment Configuration**: Uses a `.env` file for secure management of API credentials and settings.

## Prerequisites
- **Python 3.8+**: Ensure Python is installed.
- **IBM Cloud Account**: Required for watsonx.ai API access.
- **Virtual Environment**: Recommended for managing dependencies.
- **Stable Internet Connection**: Needed for fetching web data and accessing watsonx.ai APIs.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/cybernerd4u/RAG-Llama-Watsonx-A-Retrieval-Augmented-Generation-System-with-IBM-watsonx.ai.git
   cd rag-llama-watsonx
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following content:
   ```
   langchain==0.3.0
   langchain-ibm==0.3.0
   langchain-community==0.3.0
   ibm-watsonx-ai==1.1.2
   unstructured==0.15.0
   chromadb==0.5.5
   python-dotenv==1.0.1
   beautifulsoup4==4.12.3
   ```
   Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   - Create a `.env` file in the project root (`D:\ML Projects\rag-llama-watsonx\.env`) with the following:
     ```
     WATSONX_APIKEY=your_watsonx_api_key_here
     WATSONX_PROJECT_ID=your_watsonx_project_id_here
     WATSONX_URL=https://us-south.ml.cloud.ibm.com
     USER_AGENT=rag-llama-watsonx-project/1.0 (Contact: your_email@example.com)
     CHROMA_TELEMETRY_ENABLED=False
     ```
   - Replace `your_watsonx_api_key_here` and `your_watsonx_project_id_here` with your IBM Cloud API key and watsonx.ai project ID.
     - Obtain the API key from [IBM Cloud API Keys](https://cloud.ibm.com/iam/apikeys).
     - Find the project ID in the watsonx.ai project dashboard.
   - Replace `your_email@example.com` with a valid email address for the `USER_AGENT` contact.

## Usage

1. **Run the Script**:
   - Ensure the virtual environment is activated:
     ```bash
     .\venv\Scripts\activate  # Windows
     ```
   - Execute the main script:
     ```bash
     python main.py
     ```

2. **Expected Output**:
   The script processes web data, creates embeddings, and answers a set of predefined queries. Example output:
   ```
   API Key: WATSONX_API_KEY
   Project ID: WATSONX_PROJECT_ID
   URL: https://us-south.ml.cloud.ibm.com
   User Agent: rag-llama-watsonx-project/1.0 (Contact: ----@examplex.com)
   Created 51 document chunks

   Query: What is watsonx?
   Answer: IBM watsonx is a platform for building, deploying, and managing AI models, focusing on generative AI and machine learning. It integrates tools like watsonx.ai for model training and watsonx.governance for AI governance. It supports enterprise use cases with scalable, secure solutions.
   ...
   ```

3. **Sample Queries**:
   The script includes the following queries to test the RAG system:
   - What is watsonx?
   - What is IBM?
   - What is IBM Cloud?
   - What are the main components of the IBM watsonx platform?
   - How does IBM Cloud support AI and machine learning workloads?
   - What is the history of IBM Cloud’s formation?
   - What types of AI models can be used with watsonx?
   - How does watsonx.governance help in managing AI?
   - What companies have used IBM Cloud for their workloads?
   - What is the difference between watsonx.ai and watsonx.data?
   - When was IBM founded, and where is it headquartered?
   - What services does IBM Cloud offer besides compute and storage?
   - How does IBM watsonx support enterprise use cases?

4. **Customize Queries**:
   - Edit the `sample_queries` list in `main.py` to add or modify questions.
   - Example:
     ```python
     sample_queries = [
         "What is the role of watsonx in AI development?",
         "How does IBM Cloud compare to other cloud providers?"
     ]
     ```

5. **Add More Data Sources**:
   - Update the `urls` list in `main.py` to include additional web pages for context:
     ```python
     urls = [
         "https://en.wikipedia.org/wiki/IBM_Watsonx",
         "https://en.wikipedia.org/wiki/IBM_Cloud",
         "https://en.wikipedia.org/wiki/IBM"
     ]
     ```

## Project Structure
```
rag-llama-watsonx/
│
├── .env                # Environment variables (API keys, URLs, USER_AGENT)
├── .gitignore          # Ignores .env, vector_store/, data/, and venv/
├── main.py             # Main script for the RAG system
├── requirements.txt    # Python dependencies
├── vector_store/       # Chroma vector store directory (auto-generated)
└── data/               # Optional directory for saving raw documents (if enabled)
```

## Technical Details
- **LLM**: Uses IBM watsonx.ai with the `meta-llama/llama-3-405b-instruct` model for generating answers.
- **Embeddings**: Employs `ibm/slate-125m-english-rtrvr` for creating document embeddings.
- **Vector Store**: Chroma 0.5.5 for storing and retrieving document embeddings.
- **Web Scraping**: `WebBaseLoader` with `beautifulsoup4` to fetch and parse web data.
- **Text Splitting**: `RecursiveCharacterTextSplitter` with a chunk size of 1000 and overlap of 200.
- **Prompt Template**: Limits answers to three sentences for clarity and conciseness.
- **Dependencies**: Managed via `requirements.txt` to ensure reproducibility.

## Troubleshooting
Common issues and their solutions:

1. **USER_AGENT Warning**:
   - **Error**: `USER_AGENT environment variable not set, consider setting it to identify your requests`.
   - **Solution**: Ensure the `.env` file exists in the project root with `USER_AGENT=rag-llama-watsonx-project/1.0 (Contact: your_email@example.com)`. Verify `load_dotenv()` is called in `main.py`. Alternatively, set it manually:
     ```bash
     set USER_AGENT=rag-llama-watsonx-project/1.0 (Contact: your_email@example.com)
     python main.py
     ```

2. **API Authentication Errors**:
   - **Error**: `Invalid API key` or `Unauthorized`.
   - **Solution**: Verify `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` in `.env`. Regenerate the API key at [IBM Cloud API Keys](https://cloud.ibm.com/iam/apikeys). Check the project ID in the watsonx.ai dashboard.

3. **Model Unavailability**:
   - **Error**: `Model 'meta-llama/llama-3-405b-instruct' is not supported`.
   - **Solution**: Confirm model availability in your region (Dallas) via the watsonx.ai Resource Hub. Try a smaller model like `meta-llama/llama-3-3-70b-instruct`:
     ```python
     model_id="meta-llama/llama-3-3-70b-instruct"
     ```

4. **Chroma Telemetry Errors**:
   - **Error**: `Failed to send telemetry event ...: capture() takes 1 positional argument but 3 were given`.
   - **Solution**: Telemetry is disabled in `main.py` with `os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"`. Ensure this line is present or add `CHROMA_TELEMETRY_ENABLED=False` to `.env`.

5. **Deprecation Warning**:
   - **Error**: `LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported`.
   - **Solution**: The `vectorstore.persist()` call has been removed in `main.py` as Chroma auto-persists documents.

6. **Network Issues**:
   - **Error**: `Failed to load web data` or API connection errors.
   - **Solution**: Ensure a stable internet connection:
     ```bash
     curl https://us-south.ml.cloud.ibm.com
     ```
   - Check for website restrictions or temporary outages.

7. **Chroma Vector Store Issues**:
   - **Error**: Permission errors or corrupted vector store.
   - **Solution**: Delete the `vector_store` directory and retry:
     ```bash
     Remove-Item -Recurse -Force .\vector_store
     python main.py
     ```

8. **Dependency Issues**:
   - **Error**: Missing or incompatible packages.
   - **Solution**: Reinstall dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Recreate the virtual environment if needed:
     ```bash
     deactivate
     Remove-Item -Recurse -Force .\venv
     python -m venv venv
     .\venv\Scripts\activate
     pip install -r requirements.txt
     ```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 style guidelines and includes relevant tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **IBM watsonx.ai**: For providing the platform and Llama 3 model.
- **LangChain**: For the RAG framework and integrations.
- **Chroma**: For the vector store implementation.
- **Wikipedia**: For providing open-source data used in this project.

## Contact
For issues or inquiries, contact Ibrahim Sanusi at [ibrahimsanusib10@gmail.com](mailto:ibrahimsanusib10@gmail.com).