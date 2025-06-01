from langchain_ollama.llms  import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever

print("Initialising Ollama...")
model = OllamaLLM(model="llama3.2")

print("Model initialisation completed")
template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""


prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


print("Entering while...")
while True:
    
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)