import sqlite3

# Conectar ao banco de dados (ou criar se não existir)
conn = sqlite3.connect('water-quality.db')

# Criar um cursor para executar comandos SQL
cursor = conn.cursor()

# Criar a tabela 'reservatorios' com os campos 'nome' e 'coordenadas'
cursor.execute('''
    CREATE TABLE IF NOT EXISTS reservatorios (
        nome TEXT,
        coordenadas TEXT
    )
''')

# Salvar (commit) as mudanças
conn.commit()

# Fechar a conexão
conn.close()

print("Banco de dados e tabela criados com sucesso!")
