import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader  

## Streamlit app
st.set_page_config(page_title="Summarize text from YT or Website")
st.title("Summarize text from URL")
st.subheader('summarize URL')

## Get the groq api key and url field
with st.sidebar:
    groq_api_key = st.text_input("Groq API key", value = "", type='password')

generic_url = st.text_input("URL", label_visibility='collapsed')

## Prompt template
prompt_template = ''' 
    provide the summary of the following content in 300 word:
    content: {text}
'''

prompt = PromptTemplate(input_variables=['text'], template = prompt_template)

if st.button("Summarize the content from YT or Website"):
    ## validate the input
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("please enter a valid URL!")
    else:
        try:
            with st.spinner("waiting..."):
                # loading the website or yt data
                llm = ChatGroq(model ='llama-3.3-70b-versatile', api_key = groq_api_key)
                if 'youtube.com' in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info = True)
                else:
                    loader = UnstructuredURLLoader(urls = [generic_url], ssl_verify = False,
                                                   headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                                                )
                docs = loader.load()
                
                ## Chain for summarization
                chain = load_summarize_chain(llm, chain_type = 'stuff', prompt = prompt)
                output_summary = chain.run(docs)
                
                st.success(output_summary)
        except Exception as e:
            st.error(f"Exception:{e}")