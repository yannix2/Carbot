import ollama

response = ollama.chat(model="mistral", messages=[
    {"role": "system", "content": "Tu es un assistant."},
    {"role": "user", "content": "Bonjour !"}
])

print(response)
