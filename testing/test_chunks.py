from modules.chunker import TextChunker

chunker = TextChunker(max_words=150, overlap=20)
chunks = chunker.load_json_folder("/Users/annpetrosiann/Desktop/RAG_Model_Template/data")

# Each chunk contains (id, text, title)
for chunk_id, text, title in chunks:
    print(f"Title: {title}")
    print(f"ID: {chunk_id}")
    print(f"Text: {text[:50]}...\n")