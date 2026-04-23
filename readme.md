<h1>Build a local RAG with Ollama</h1>

<h2>Watch the full tutorial on my YouTube Channel</h2>
<div>

<a href="https://youtu.be/c5jHhMXmXyo">
    <img src="thumbnail_small.png" alt="Thomas Janssen Youtube" width="200"/>
</a>
</div>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/ThomasJanssen-tech/Local-RAG-with-Ollama
cd Local-RAG-With-Ollama
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>4. Install libraries</h3>

```
pip install -r requirements.txt
```

<h3>5. Add Bright Data API Key</h3>
<ul>
<li>Get your $15 Bright Data credits: https://brdta.com/tomstechacademy</li>
<li>Rename the .env.example file to .env</li>
<li>Add your Bright Data API key</li>
<li><i>If you want to use ChatGPT or Anthropic models, add an API key (not required for Ollama)</i></li>
</ul>

<h2>Executing the scripts</h2>

- Open a terminal in VS Code

- Execute the following command:

```
python run 1_scraping_wikipedia.py
python run 2_chunking_embedding_ingestion.py
streamlit run 3_chatbot.py
```

<h2>Further reading</h2>
<ul>
<li>https://www.ibm.com/think/topics/vector-embedding</li>
<li>https://ollama.com/blog/embedding-models</li>
<li>https://python.langchain.com/docs/integrations/vectorstores/chroma/</li>
<li>https://python.langchain.com/docs/integrations/text_embedding/ollama/</li>
<li>https://ollama.com/library/mxbai-embed-large</li>
<li>https://ollama.com/library/qwen3</li>
</ul>
