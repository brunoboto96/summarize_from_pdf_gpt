### Summarize_Text_with_GPT

ggit@github.com:brunoboto96/summarize_from_pdf_gpt.git

## RUN code

pip intall -r requirements.txt

- add a document.pdf not sure if it needs to be clean of images, might need to automate this and ignore images, as it generates lots of ascii code, or maybe not. Didnt check.
- run: git update-index --assume-unchanged api_key.py
- add your API key to api_key.py

python main.py

Run once then:
uncomment line 42 to initiate summary
