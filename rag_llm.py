from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.llms import HuggingFacePipeline
import torch
from langchain_community.llms import CTransformers
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_qdrant import FastEmbedSparse, RetrievalMode



class RAGPipelineSetup:
    def __init__(self, qdrant_url, qdrant_api_key, qdrant_collection_name, huggingface_api_key, embeddings_model_name, groq_api_key,groq_api_key2,groq_api_key3, reranker_model_name):
        self.QDRANT_URL = qdrant_url
        self.QDRANT_API_KEY = qdrant_api_key
        self.QDRANT_COLLECTION_NAME = qdrant_collection_name
        self.HUGGINGFACE_API_KEY = huggingface_api_key
        self.EMBEDDINGS_MODEL_NAME = embeddings_model_name
        self.GROQ_API_KEY = groq_api_key
        self.GROQ_API_KEY2 = groq_api_key2
        self.GROQ_API_KEY3 = groq_api_key3
        self.RERANKER_MODEL_NAME = reranker_model_name
        self.current_source = None
        self.rag_pipeline = None
        self.embeddings = self.load_embeddings()
        self.retriever = None
        self.pipe = None
        self.prompt = self.load_prompt_template()

    def load_embeddings(self):
        bge_embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name=self.EMBEDDINGS_MODEL_NAME,
            api_key=self.HUGGINGFACE_API_KEY,
            model_kwargs={'device': 'auto'}
        )
        llm = ChatGroq(
         temperature=0, 
         groq_api_key=self.GROQ_API_KEY, 
         model_name="gemma2-9b-it"
         )
        embeddings = HypotheticalDocumentEmbedder.from_llm(llm, bge_embeddings, prompt_key="web_search")
        return embeddings

    def load_retriever(self, retriever_name, embeddings):
      # Tạo client kết nối với Qdrant
      client = QdrantClient(
          url=self.QDRANT_URL,
          api_key=self.QDRANT_API_KEY,
          prefer_grpc=False
      )
      sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

      # Tạo đối tượng Qdrant với collection và embeddings
      db = QdrantVectorStore(
          client=client,
          embedding=embeddings,
          sparse_embedding=sparse_embeddings,
          sparse_vector_name="sparse_vector",
          collection_name=self.QDRANT_COLLECTION_NAME,
          retrieval_mode=RetrievalMode.HYBRID
      )

      # Tạo retriever từ Qdrant với top k kết quả ban đầu
      base_retriever = db.as_retriever(search_kwargs={"k": 5},search_type="mmr")

    #   # Tạo mô hình reranker với HuggingFaceCrossEncoder
    #   model = HuggingFaceCrossEncoder(model_name=self.RERANKER_MODEL_NAME)
    #   reranker = CrossEncoderReranker(model=model, top_n=5)

      # Tạo LongContextReorder
      reordering = LongContextReorder()

      # Kết hợp reranker và reordering thành một pipeline
      compressor = DocumentCompressorPipeline(
          transformers=[reordering]  #Khi nào dùng thì thêm reranker vô
      )

      # Tạo retriever với contextual compression sử dụng pipeline compressor
      compression_retriever = ContextualCompressionRetriever(
          base_compressor=compressor,
          base_retriever=base_retriever
      )

      return compression_retriever

    def load_model_pipeline(self, max_new_tokens=1024):
        llm = ChatGroq(
            temperature=0, 
            groq_api_key=self.GROQ_API_KEY2, 
            model_name="llama3-groq-70b-8192-tool-use-preview"
        )
        return llm

    def load_prompt_template(self):
        query_template = '''Bạn là trợ lý ảo thông minh hỗ trợ giải đáp câu hỏi cho sinh viên, 
                            hãy dựa vào context được cung cấp và 1 ít kiến thức của bạn để trả lời câu hỏi của người dùng bằng Tiếng Việt (context),
                            nếu bạn không có câu trả lời hãy gợi ý cách tìm ra được thông tin.
                            Hãy chọn lọc thông tin để cung cấp câu trả lời ngắn gọn nhưng đầy đủ cho người dùng
                            Trong văn bản được cung cấp sẽ có những từ được nối bằng _ nhưng bạn không cần giữ nguyên ký tự _  
                            Trong câu trả lời của bạn không được dùng các từ như "từ văn bản được cung cấp, "văn bản được đề cập", ...
                            \n### Context:{context} \n\n### Người dùng: {question}'''
        prompt = PromptTemplate(template=query_template, input_variables=["context", "question"])
        return prompt

    def load_rag_pipeline(self, llm, retriever, prompt):
        # Tạo QA chain với load_qa_chain
        qa_chain = load_qa_chain(llm, chain_type="stuff",prompt=prompt)  
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        _template ="""Bạn là một trợ lý thông minh. Nhiệm vụ của bạn là viết lại câu hỏi hiện tại một cách rõ ràng và dễ hiểu nhất có thể. Hãy ưu tiên giữ nguyên nội dung chính của câu hỏi hiện tại và chỉ sử dụng ngữ cảnh từ lịch sử trò chuyện nếu nó giúp làm rõ thêm ý nghĩa của câu hỏi. Không tự động thay đổi câu hỏi hiện tại dựa trên ngữ cảnh trước trừ khi thực sự cần thiết.

        ### Câu hỏi hiện tại:
        {question}

        ### Lịch sử cuộc trò chuyện (nếu cần):
        {chat_history}

        Viết lại câu hỏi hiện tại sao cho ngắn gọn, rõ ràng,bằng tiếng việt nhưng vẫn giữ nguyên ý nghĩa chính của câu hỏi gốc. Chỉ sử dụng lịch sử cuộc trò chuyện nếu nó thực sự cần thiết để làm rõ câu hỏi của người dùng.

        """
        condense_question_prompt_template = PromptTemplate.from_template(_template)

        LLM = ChatGroq(
            temperature=0, 
            groq_api_key=self.GROQ_API_KEY3, 
            model_name="gemma2-9b-it"
        )
        question_generator = LLMChain(llm=LLM, prompt=condense_question_prompt_template, memory=memory)
        # Tạo ConversationalRetrievalChain với các tham số cần thiết
        conversational_chain = ConversationalRetrievalChain(
            retriever=retriever,
            combine_docs_chain=qa_chain,   # Chuỗi kết hợp tài liệu với mô hình LLM
            question_generator=question_generator,       
            return_source_documents=False,
            memory=memory     
        )
        return conversational_chain

    def rag(self, source):
        if source == self.current_source:
            return self.rag_pipeline
        else:
            self.retriever = self.load_retriever(retriever_name=source, embeddings=self.embeddings)
            self.pipe = self.load_model_pipeline()
            self.rag_pipeline = self.load_rag_pipeline(llm=self.pipe, retriever=self.retriever, prompt=self.prompt)
            self.current_source = source
            return self.rag_pipeline
