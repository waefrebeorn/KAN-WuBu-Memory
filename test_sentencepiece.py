import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("C:/Projects/KAN-WuBu-Memory/models/Llama_32_1B/tokenizer.model")
print("Tokenizer loaded successfully.")
