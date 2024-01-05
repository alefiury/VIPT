from data.text.symbols import symbols

def symbols_to_ids(text):
    return [symbols.index(c) for c in text]

def ids_to_symbols(ids):
    return [symbols[id] for id in ids]