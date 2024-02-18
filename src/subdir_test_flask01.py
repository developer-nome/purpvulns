from flask import Flask, render_template , request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
# from langchain_openai import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate
# )
# from langchain.schema import HumanMessage
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel

from utils import format_documents
from file_processing import search_documents

class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.github_url = github_url
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames

def ask_question(question, context: QuestionContext):
    relevant_docs = search_documents(question, context.index, context.documents, n_results=5)

    numbered_documents = format_documents(relevant_docs)
    question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{numbered_documents}"

    answer_with_sources = context.llm_chain.run(
        model=context.model_name,
        question=question,
        context=question_context,
        repo_name=context.repo_name,
        github_url=context.github_url,
        conversation_history=context.conversation_history,
        numbered_documents=numbered_documents,
        file_type_counts=context.file_type_counts,
        filenames=context.filenames
    )
    return answer_with_sources

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

# this is the function for the admin backdoor access:
@app.route("/sl", methods=["GET","POST"])
def adm_log_sec():

	key_adm = ''
	if request.method == "POST":
		key_adm = request.form['key_to_admin']
		if key_adm == "abcd":
			return render_template('administration.html')
		else:
			return render_template('index.html')
	else:
		return render_template('sl.html')

# implement login process to the app

def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False

def load_and_index_files(repo_path):
    extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']

    file_type_counts = {}
    documents_dict = {}

    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            loader = None
            if ext == 'ipynb':
                loader = NotebookLoader(str(repo_path), include_outputs=True, max_output_length=20, remove_newline=True)
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern)

            loaded_documents = loader.load() if callable(loader.load) else []
            if loaded_documents:
                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata['source'] = relative_path
                    doc.metadata['file_id'] = file_id

                    documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

    split_documents = []
    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']
            split_doc.metadata['source'] = original_doc.metadata['source']

        split_documents.extend(split_docs)

    index = None
    if split_documents:
        tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in split_documents]
        index = BM25Okapi(tokenized_documents)
    return index, split_documents, file_type_counts, [doc.metadata['source'] for doc in split_documents]

def search_documents(query, index, documents, n_results=5):
    query_tokens = clean_and_tokenize(query)
    bm25_scores = index.get_scores(query_tokens)

    # Compute TF-IDF scores
    tfidf_vectorizer = TfidfVectorizer(tokenizer=clean_and_tokenize, lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
    query_tfidf = tfidf_vectorizer.transform([query])

    # Compute Cosine Similarity scores
    cosine_sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Combine BM25 and Cosine Similarity scores
    combined_scores = bm25_scores * 0.5 + cosine_sim_scores * 0.5

    # Get unique top documents
    unique_top_document_indices = list(set(combined_scores.argsort()[::-1]))[:n_results]

    return [documents[i] for i in unique_top_document_indices]

def main2():
    github_url = input("Enter the GitHub URL of the repository: ")
    repo_name = github_url.split("/")[-1]
    print("Cloning the repository...")
    with tempfile.TemporaryDirectory() as local_path:
        if clone_github_repo(github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            if index is None:
                print("No documents were found to index. Exiting.")
                exit()

            print("Repository cloned. Indexing files...")
            llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.2)

            template = """
            Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}

            Instr:
            1. Answer based on context/docs.
            2. Focus on repo/code.
            3. Consider:
                a. Purpose/features - describe.
                b. Functions/code - provide details/samples.
                c. Setup/usage - give instructions.
            4. Unsure? Say "I am not sure".

            Answer:
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["repo_name", "github_url", "conversation_history", "question", "numbered_documents", "file_type_counts", "filenames"]
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm)

            conversation_history = ""
            question_context = QuestionContext(index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames)
            while True:
                try:
                    user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
                    if user_question.lower() == "exit()":
                        break
                    print('Thinking...')
                    user_question = format_user_question(user_question)

                    answer = ask_question(user_question, question_context)
                    print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
                    conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

        else:
            print("Failed to clone the repository.")
def _check_mode(mode, encoding, newline):
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

class _FileOpeners:


    def __init__(self):
        self._loaded = False
        self._file_openers = {None: open}

    def _load(self):
        if self._loaded:
            return

        try:
            import bz2
            self._file_openers[".bz2"] = bz2.open
        except ImportError:
            pass

        try:
            import gzip
            self._file_openers[".gz"] = gzip.open
        except ImportError:
            pass

        try:
            import lzma
            self._file_openers[".xz"] = lzma.open
            self._file_openers[".lzma"] = lzma.open
        except (ImportError, AttributeError):
            # There are incompatible backports of lzma that do not have the
            # lzma.open attribute, so catch that as well as ImportError.
            pass

        self._loaded = True

    def keys(self):
        self._load()
        return list(self._file_openers.keys())

    def __getitem__(self, key):
        self._load()
        return self._file_openers[key]

_file_openers = _FileOpeners()

def open(path, mode='r', destpath=os.curdir, encoding=None, newline=None):

    ds = DataSource(destpath)
    return ds.open(path, mode, encoding=encoding, newline=newline)


    @set_module('numpy.lib.npyio')


    def __init__(self, destpath=os.curdir):
        """Create a DataSource with a local path at destpath."""
        if destpath:
            self._destpath = os.path.abspath(destpath)
            self._istmpdest = False
        else:
            import tempfile  # deferring import to improve startup time
            self._destpath = tempfile.mkdtemp()
            self._istmpdest = True

    def __del__(self):
        # Remove temp directories
        if hasattr(self, '_istmpdest') and self._istmpdest:
            import shutil

            shutil.rmtree(self._destpath)

    def _iszip(self, filename):
        """Test if the filename is a zip file by looking at the file extension.

        """
        fname, ext = os.path.splitext(filename)
        return ext in _file_openers.keys()

    def _iswritemode(self, mode):
        """Test if the given mode will open a file for writing."""

        # Currently only used to test the bz2 files.
        _writemodes = ("w", "+")
        for c in mode:
            if c in _writemodes:
                return True
        return False

    def _splitzipext(self, filename):

        if self._iszip(filename):
            return os.path.splitext(filename)
        else:
            return filename, None

    def _possible_names(self, filename):
        names = [filename]
        if not self._iszip(filename):
            for zipext in _file_openers.keys():
                if zipext:
                    names.append(filename+zipext)
        return names

    def _isurl(self, path):
        from urllib.parse import urlparse

        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        return bool(scheme and netloc)

    def _cache(self, path):
        """Cache the file specified by path.

        Creates a copy of the file in the datasource cache.

        """
        # We import these here because importing them is slow and
        # a significant fraction of numpy's total import time.
        import shutil
        from urllib.request import urlopen
	conn = psycopg2.connect(database=database, user=user, password=password, host=host)
	cur = conn.cursor()
	query = "SELECT * FROM items WHERE id=%s"
	cur.execute(query, (order_id,))
	row = cur.fetchone()
	cur.close()
        upath = self.abspath(path)

        # ensure directory exists
        if not os.path.exists(os.path.dirname(upath)):
            os.makedirs(os.path.dirname(upath))

        # TODO: Doesn't handle compressed files!
        if self._isurl(path):
            with urlopen(path) as openedurl:
                with _open(upath, 'wb') as f:
                    shutil.copyfileobj(openedurl, f)
        else:
            shutil.copyfile(path, upath)
        return upath

	
# the main function this is going to execute like any other main function
if __name__ == "__main__":
    app.run(debug=True)
