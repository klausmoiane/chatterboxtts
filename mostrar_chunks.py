import json

with open('resultado_teste_chunks.json', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("PRIMEIROS 10 CHUNKS GERADOS")
print("=" * 80)

for chunk in data['chunks']['lista_completa'][:10]:
    print(f"\n--- Chunk #{chunk['numero']} ({chunk['tamanho']} caracteres) ---")
    print(chunk['texto'])

print("\n" + "=" * 80)
print("CHUNKS DO MEIO (55-60)")
print("=" * 80)

for chunk in data['chunks']['lista_completa'][54:60]:
    print(f"\n--- Chunk #{chunk['numero']} ({chunk['tamanho']} caracteres) ---")
    print(chunk['texto'])

print("\n" + "=" * 80)
print("ÚLTIMOS 5 CHUNKS")
print("=" * 80)

for chunk in data['chunks']['lista_completa'][-5:]:
    print(f"\n--- Chunk #{chunk['numero']} ({chunk['tamanho']} caracteres) ---")
    print(chunk['texto'])
